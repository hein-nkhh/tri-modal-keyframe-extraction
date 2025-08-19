import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
from multiprocessing import Pool
from threading import Thread
from queue import Queue

def get_videos(directory='../../test_videos'):
    """
    Retrieves video file paths from the specified directory.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    video_files = []
    # Walk through the directory and collect video files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    # Check if any video files were found
    if not video_files:
        print(f"No video files found in '{directory}'. Please add videos to process.")

    return video_files

def initialize_model():
    """
    Initializes the ResNet-50 v1.5 model with IMAGENET1K_V2 weights for feature extraction.
    """
    from torchvision.models import resnet50, ResNet50_Weights

    # Load ResNet-50 v1.5 model with pre-trained weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()  # Set the model to evaluation mode

    # Remove the last fully connected layer to get feature vectors
    modules = list(model.children())[:-1]  # Exclude the final FC layer
    model = torch.nn.Sequential(*modules)

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def preprocess_frames(frames):
    """
    Preprocesses a list of frames for input into ResNet-50 on GPU.

    Parameters:
    - frames: List of frames (numpy arrays in BGR format)

    Returns:
    - Preprocessed batch tensor ready for feature extraction
    """
    preprocess = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),  # Ensure images are float32
        transforms.Resize(256),                       # Resize shortest side to 256 pixels
        transforms.CenterCrop(224),                   # Center crop to 224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                             std=[0.229, 0.224, 0.225]),  # and standard deviation
    ])

    # Convert list of frames (numpy arrays) to a single numpy array
    frames = np.stack(frames, axis=0)  # Shape: (batch_size, height, width, channels)

    # Convert to torch tensor and rearrange dimensions to match model expectations
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).cuda(non_blocking=True)  # Shape: (batch_size, channels, height, width)

    # Apply preprocessing transforms (on GPU for speed)
    frames = preprocess(frames)
    return frames

def extract_features(model, frames_batch):
    """
    Extracts features from a batch of preprocessed frames using the model.

    Parameters:
    - model: The pre-trained ResNet-50 model
    - frames_batch: Batch of preprocessed frames (torch tensor on GPU)

    Returns:
    - Numpy array of feature vectors
    """
    with torch.no_grad():
        # Forward pass through the model to get features
        features = model(frames_batch)
        # Flatten features and move to CPU
        features = features.view(features.size(0), -1).cpu().numpy()
    return features

def frame_reader(video_file, frame_queue, frame_interval):
    """
    Reads frames from the video file and pushes them to the queue.

    Parameters:
    - video_file: Path to the video file
    - frame_queue: Queue to put frames into
    - frame_interval: Interval at which to sample frames
    """
    cap = cv2.VideoCapture(video_file)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Sample frames at specified intervals
        if frame_idx % frame_interval == 0:
            frame_queue.put((frame_idx, frame))
        frame_idx += 1
    cap.release()
    # Signal that there are no more frames
    frame_queue.put(None)

def process_video(video_file, batch_size, frame_interval, model):
    """
    Processes a single video, extracting features for specified frames.

    Parameters:
    - video_file: Path to the video file
    - batch_size: Number of frames to process in a batch
    - frame_interval: Interval at which to sample frames
    - model: The pre-trained ResNet-50 model

    Returns:
    - Tuple containing the video name and a list of (frame_id, feature_vector) tuples
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    # Create a queue for communication between reader thread and main thread
    frame_queue = Queue(maxsize=batch_size * 2)  # Buffer size to prevent blocking

    # Start a separate thread for reading frames
    reader_thread = Thread(target=frame_reader, args=(video_file, frame_queue, frame_interval))
    reader_thread.start()

    features_list = []
    frames = []
    frame_ids = []

    while True:
        item = frame_queue.get()
        if item is None:
            # No more frames to process
            break
        frame_idx, frame = item
        frames.append(frame)
        frame_ids.append(frame_idx)

        if len(frames) == batch_size:
            # Preprocess batch of frames
            batch_tensor = preprocess_frames(frames)
            # Extract features
            features = extract_features(model, batch_tensor)
            # Store features with corresponding frame IDs
            for idx, feature_vector in enumerate(features):
                features_list.append((frame_ids[idx], feature_vector))
            # Clear the lists for the next batch
            frames = []
            frame_ids = []

    # Process any remaining frames
    if frames:
        batch_tensor = preprocess_frames(frames)
        features = extract_features(model, batch_tensor)
        for idx, feature_vector in enumerate(features):
            features_list.append((frame_ids[idx], feature_vector))

    # Wait for the reader thread to finish
    reader_thread.join()
    return video_name, features_list

def process_videos_on_gpu(gpu_id, video_files, batch_size, frame_interval):
    """
    Processes a subset of videos on a specific GPU.

    Parameters:
    - gpu_id: ID of the GPU to use
    - video_files: List of video files to process
    - batch_size: Number of frames to process in a batch
    - frame_interval: Interval at which to sample frames

    Returns:
    - Dictionary with video names as keys and lists of (frame_id, feature_vector) tuples as values
    """
    # Set the device to the specified GPU
    torch.cuda.set_device(gpu_id)
    # Initialize the model on the specified GPU
    model = initialize_model()

    features_dict = {}
    for video_file in tqdm(video_files, desc=f"GPU {gpu_id}"):
        try:
            # Process the video and extract features
            video_name, features_list = process_video(video_file, batch_size, frame_interval, model)
            features_dict[video_name] = features_list
        except Exception as e:
            print(f"Error processing {video_file} on GPU {gpu_id}: {e}")

    return features_dict

def save_features_dict_hdf5(features_dict, filename):
    """
    Saves the features_dict to an HDF5 file for future use.

    Parameters:
    - features_dict: Dictionary with video names as keys and lists of (frame_id, feature_vector) tuples as values
    - filename: Name of the HDF5 file to save the features_dict
    """
    with h5py.File(filename, 'w') as h5f:
        for video_name, features_list in features_dict.items():
            # Extract frame IDs and feature vectors
            frame_ids = [item[0] for item in features_list]
            features_array = np.array([item[1] for item in features_list])
            # Create a group for each video
            grp = h5f.create_group(video_name)
            # Save frame IDs and features
            grp.create_dataset('frame_ids', data=frame_ids)
            grp.create_dataset('features', data=features_array)
    print(f"Features dictionary saved to {filename}")

def extract_image_features(video_path = '../../test_videos', frame_interval = 10):
    # Get list of video files to process
    video_files = get_videos(video_path)
    if not video_files:
        return

    batch_size = 256

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # Split the video files among the available GPUs
        video_chunks = np.array_split(video_files, num_gpus)
        args = [(i, video_chunks[i], batch_size, frame_interval) for i in range(num_gpus)]

        # Create a multiprocessing pool to process videos on multiple GPUs
        with Pool(num_gpus) as pool:
            results = pool.starmap(process_videos_on_gpu, args)

        # Combine the results from all GPUs
        features_dict = {}
        for gpu_result in results:
            features_dict.update(gpu_result)
    else:
        # Only one GPU available, process all videos on GPU 0
        features_dict = process_videos_on_gpu(0, video_files, batch_size, frame_interval)

    # Save the extracted features to an HDF5 file
    save_features_dict_hdf5(features_dict, filename='feature_extraction/image_features/image_features_dict.h5')

if __name__ == "__main__":
    extract_image_features()
