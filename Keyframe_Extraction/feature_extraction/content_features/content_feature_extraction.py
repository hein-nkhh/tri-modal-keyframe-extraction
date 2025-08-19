import os
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import csv
from multiprocessing import Pool, set_start_method
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from feature_extraction.content_features import content_embeddings


def get_videos(directory='../../test_videos'):
    """
    Retrieves video file paths from the specified directory.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"No video files found in '{directory}'. Please add videos to process.")

    return video_files


def initialize_model():
    """
    Initializes the Llama 3.2-11B-Vision-Instruct model and processor with quantization.
    """
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Set quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model with quantization
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map=None
    ).to('cuda')
    model.eval()

    # Compile the model for potential speed gains (requires Torch 2.0+)
    model = torch.compile(model, mode="default")

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, torch.device('cuda')


def generate_description(frames, model, processor, device):
    """
    Generates content descriptions for a list of frames using the model.
    If the model returns a response indicating it does not see the image,
    we replace the description with a default placeholder.
    """
    # Convert each frame from BGR to RGB using slicing (usually faster than cv2.cvtColor)
    images = [Image.fromarray(frame[..., ::-1]) for frame in frames]

    # Since the prompt is identical for every image, compute it once:
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                       {"type": "text", "text": "In one sentence, describe the visible content of this provided image: "}] }],
        add_generation_prompt=True
    )
    # Replicate the same prompt for each image (batch_size remains 1 for each call)
    input_texts = [prompt] * len(images)

    inputs = processor(
        images=images,
        text=input_texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True
    ).to(device)

    prompt_lengths = inputs['input_ids'].ne(processor.tokenizer.pad_token_id).sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,  # greedy decoding
            temperature=None,
            top_p=None
        )

    descriptions = []
    unwanted_tokens = ["<|eot_id|>", "<|finetune_right_pad_id|>"]

    # Keywords that indicate the model did not "see" the image
    no_image_keywords = [
        "have access to an image",
        "no image provided",
        "no image",
        "not seeing",
        "image is blank",
        "share the image with me",
        "see an image",
        "access images",
        "access the images",
        "unable to view or describe",
        "not able to view the image",
        "to view or access images",
        "see the image",
        "Unfortunately",
    ]

    for i, output in enumerate(outputs):
        prompt_length = prompt_lengths[i]
        response = processor.decode(output[prompt_length:], skip_special_tokens=True)
        # Remove unwanted tokens manually
        for t in unwanted_tokens:
            response = response.replace(t, "")
        response = response.strip()

        if any(keyword in response.lower() for keyword in no_image_keywords):
            response = "No visible content"

        descriptions.append(response)

    return descriptions


def process_videos_on_gpu(args):
    """
    Processes the assigned videos on the given GPU.
    Uses a background thread to prefetch frames while running inference.
    """
    import threading
    from queue import Queue, Empty

    gpu_id, video_files, frame_interval = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    model, processor, device = initialize_model()

    all_results = []

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []

        pbar = tqdm(total=total_frames, desc=f"Processing {video_name} on GPU {gpu_id}", leave=False)

        # Queue to hold prefetched frames (adjust maxsize as needed)
        frame_queue = Queue(maxsize=20)
        stop_event = threading.Event()

        def read_frames():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    frame_queue.put((idx, frame))
                idx += 1
                # Update progress bar as each frame is read.
                pbar.update(1)
            stop_event.set()

        reader_thread = threading.Thread(target=read_frames)
        reader_thread.start()

        # Process frames as they become available.
        while not (stop_event.is_set() and frame_queue.empty()):
            try:
                frame_idx, frame = frame_queue.get(timeout=1)
            except Empty:
                continue
            try:
                # Process a single frame (batch size = 1)
                description = generate_description([frame], model, processor, device)[0]
                results.append({
                    'video_id': video_name,
                    'frame_id': frame_idx,
                    'description': description
                })
            except Exception as e:
                print(f"Error processing frame {frame_idx} of {video_name} on GPU {gpu_id}: {e}")

        reader_thread.join()
        pbar.close()
        cap.release()
        all_results.extend(results)

    del model
    torch.cuda.empty_cache()

    return all_results


def extract_content_features(video_path='../../test_videos', frame_interval=10):
    """
    Retrieves videos, processes them in parallel across available GPUs, and saves
    the frame descriptions to CSV. Finally, converts the descriptions to embeddings.
    """
    # Ensure multiprocessing uses 'spawn' method
    set_start_method('spawn', force=True)

    # Step 1: Get list of video files
    video_files = get_videos(video_path)
    if not video_files:
        return

    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return

    # Distribute videos across GPUs
    video_chunks = [[] for _ in range(num_gpus)]
    for idx, video_file in enumerate(video_files):
        gpu_id = idx % num_gpus
        video_chunks[gpu_id].append(video_file)

    # Prepare arguments for multiprocessing
    args_list = []
    for gpu_id in range(num_gpus):
        args_list.append((gpu_id, video_chunks[gpu_id], frame_interval))

    # Use a multiprocessing Pool to process videos in parallel
    with Pool(processes=num_gpus) as pool:
        results = pool.map(process_videos_on_gpu, args_list)

    # Combine results from all GPUs
    all_results = []
    for gpu_results in results:
        all_results.extend(gpu_results)

    # Save the results to a CSV file
    output_csv = 'feature_extraction/content_features/content_descriptions.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video_id', 'frame_id', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"Content descriptions saved to {output_csv}")

    print("Converting descriptions into embeddings...")
    content_embeddings.sentence_to_embeddings()


if __name__ == "__main__":
    extract_content_features()
