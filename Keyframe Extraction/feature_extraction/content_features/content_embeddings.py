import csv
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def sentence_to_embeddings():
    # 1. Initialize the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # 2. Read the CSV file with columns: video_id, frame_id, description
    csv_file_path = "feature_extraction/content_features/content_descriptions.csv"
    data = {}

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            frame_id = row['frame_id']
            description = row['description']

            if video_id not in data:
                data[video_id] = {"frame_ids": [], "descriptions": []}
            data[video_id]["frame_ids"].append(frame_id)
            data[video_id]["descriptions"].append(description)

    # 3. Process descriptions and encode them into embeddings
    batch_size = 64
    embeddings_dict = {}

    for video_id, content in tqdm(data.items(), desc="Encoding descriptions by video"):
        frame_ids = content["frame_ids"]
        descriptions = content["descriptions"]

        # Encode descriptions in batches
        embeddings_list = []
        for start_idx in range(0, len(descriptions), batch_size):
            end_idx = start_idx + batch_size
            batch_descriptions = descriptions[start_idx:end_idx]
            batch_embeddings = model.encode(batch_descriptions, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)

        # Concatenate all batch embeddings for the video
        embeddings = np.concatenate(embeddings_list, axis=0)
        embeddings_dict[video_id] = {"frame_ids": frame_ids, "features": embeddings}

    # 4. Save embeddings to an HDF5 file with the desired structure
    h5_output_path = "feature_extraction/content_features/content_description_features.h5"
    with h5py.File(h5_output_path, 'w') as hf:
        for video_id, content in embeddings_dict.items():
            video_group = hf.create_group(video_id)

            # Save frame IDs as a dataset
            video_group.create_dataset("frame_ids", data=np.array(content["frame_ids"], dtype='S'))

            # Save features as another dataset
            video_group.create_dataset("features", data=content["features"])

    print(f"Saved embeddings to feature_extraction/content_features/{h5_output_path}")

if __name__ == "__main__":
    sentence_to_embeddings()
