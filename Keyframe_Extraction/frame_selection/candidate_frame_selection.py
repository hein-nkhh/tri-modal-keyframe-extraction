import os
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from frame_selection import keyframe_selection

def load_fused_features(fused_file):
    """
    Loads the fused features and the vid_frame_map from an HDF5 file.

    Expects two datasets:
      - 'fused_features': shape (N, d)
      - 'vid_frame_map': shape (N,), each entry is a string "video_id frame_id"

    Returns:
        fused_features : np.ndarray of shape (N, d)
        vid_frame_map : list of tuples (video_id, frame_id)
    """
    if not os.path.isfile(fused_file):
        raise FileNotFoundError(f"Fused feature file not found: {fused_file}")

    with h5py.File(fused_file, 'r') as f:
        if 'fused_features' not in f:
            raise KeyError("'fused_features' dataset not found in HDF5.")
        fused_features = f['fused_features'][:]

        if 'vid_frame_map' not in f:
            raise KeyError("'vid_frame_map' dataset not found in HDF5.")
        raw_map = f['vid_frame_map'][:]

    vid_frame_map = []
    for item in raw_map:
        s = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        video_id, frame_id_str = s.split()
        frame_id = int(frame_id_str)
        vid_frame_map.append((video_id, frame_id))
    return fused_features, vid_frame_map


def load_clusters(clusters_file):
    """
    Loads clustering results from an HDF5 file structured as follows:
      For each video_id group:
         - 'frame_ids': dataset of frame_ids (e.g., [23, 24, ...])
         - 'cluster_labels': dataset of cluster labels (same length as frame_ids)

    Returns:
        clusters_dict : dict
            { video_id : { 'frame_ids': np.array, 'cluster_labels': np.array } }
    """
    if not os.path.isfile(clusters_file):
        raise FileNotFoundError(f"Clusters file not found: {clusters_file}")

    clusters_dict = {}
    with h5py.File(clusters_file, 'r') as f:
        for video_id in f.keys():
            grp = f[video_id]
            if 'frame_ids' not in grp or 'cluster_labels' not in grp:
                print(f"Warning: video {video_id} missing 'frame_ids' or 'cluster_labels'. Skipping.")
                continue
            frame_ids = grp['frame_ids'][:]
            cluster_labels = grp['cluster_labels'][:]
            clusters_dict[video_id] = {
                'frame_ids': frame_ids,
                'cluster_labels': cluster_labels
            }
    return clusters_dict


def find_medoid_candidate(features):
    """
    Given a 2D array of features for frames in one cluster,
    computes pairwise distances and returns the index of the medoid,
    i.e., the index of the feature vector with the smallest total distance
    to all other vectors.

    Parameters:
        features : np.ndarray, shape (n, d)

    Returns:
        medoid_idx : int (0-based index within 'features')
    """
    if features.shape[0] == 1:
        return 0
    # Compute pairwise Euclidean distances
    D = pairwise_distances(features, metric='euclidean')
    total_dists = D.sum(axis=1)
    medoid_idx = np.argmin(total_dists)
    return medoid_idx


def find_candidate_keyframes(video_folder='../../test_videos'):
    # Define file paths (adjust these paths as needed)
    fused_file = "feature_extraction/cross_modal_features/fused_features.h5"
    clusters_file = "clustering/clusters_per_video.h5"
    output_csv = "frame_selection/candidate_keyframes.csv"

    # Load fused features and global mapping (each row corresponds to a frame)
    print("Loading fused features and mapping...")
    fused_features, vid_frame_map = load_fused_features(fused_file)
    total_frames = fused_features.shape[0]
    print(f"Loaded {total_frames} fused feature vectors.")

    # Build a dictionary mapping (video_id, frame_id) -> global index.
    mapping_dict = {(vid, fid): idx for idx, (vid, fid) in enumerate(vid_frame_map)}

    # Load per-video clustering results.
    print("Loading clustering results...")
    clusters_dict = load_clusters(clusters_file)

    # For each video, select a candidate keyframe per cluster via medoid selection.
    candidate_keyframes = []  # List of dicts: { 'video_id': str, 'cluster_label': int, 'candidate_frame_id': int }
    for video_id, cluster_data in clusters_dict.items():
        frame_ids = cluster_data['frame_ids']  # frames in that video (as stored originally)
        cluster_labels = cluster_data['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        # Process each cluster (ignoring outliers with label -1)
        for cl in unique_clusters:
            if cl == -1:
                continue  # Skip outliers.
            # Get frame IDs belonging to this cluster.
            cluster_mask = (cluster_labels == cl)
            cluster_frame_ids = frame_ids[cluster_mask]
            # Use mapping_dict to get global indices for these frames.
            global_indices = []
            for fid in cluster_frame_ids:
                key = (video_id, fid)
                if key in mapping_dict:
                    global_indices.append(mapping_dict[key])
                else:
                    print(f"Warning: Mapping for ({video_id}, {fid}) not found.")
            if len(global_indices) == 0:
                continue
            # Extract fused features for frames in this cluster.
            cluster_feats = fused_features[global_indices, :]  # shape (n, d)
            # Compute medoid index.
            medoid_local_idx = find_medoid_candidate(cluster_feats)
            candidate_frame_id = cluster_frame_ids[medoid_local_idx]
            candidate_keyframes.append({
                'video_id': video_id,
                'cluster_label': int(cl),
                'candidate_frame_id': int(candidate_frame_id)
            })

    # Save candidate keyframes to CSV.
    print(f"Saving candidate keyframes to frame_selection/{output_csv} ...")
    df = pd.DataFrame(candidate_keyframes)
    df.to_csv(output_csv, index=False)
    print("Candidate keyframes saved.")

    print("Selecting the Final Keyframes...")
    keyframe_selection.find_keyframes(video_folder)



if __name__ == "__main__":
    find_candidate_keyframes()
