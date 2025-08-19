import os
import h5py
import numpy as np
import hdbscan
import gc
import random
import umap
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_fused_features(h5_path):
    """
    Loads the fused features and the vid_frame_map from an HDF5 file.
    Expects two datasets:
      1) 'fused_features': shape (N, d)
      2) 'vid_frame_map': shape (N,), each entry is a string "video_id frame_id"
    Returns
    -------
    fused_features : np.ndarray of shape (N, d)
    vid_frame_map : list of tuples (video_id, frame_id)
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Fused feature file not found: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'fused_features' not in f:
            raise KeyError("'fused_features' dataset not found in HDF5.")
        fused_features = f['fused_features'][:]

        if 'vid_frame_map' not in f:
            raise KeyError("'vid_frame_map' dataset not found in HDF5.")
        raw_map = f['vid_frame_map'][:]

    vid_frame_map = []
    for item in raw_map:
        # Convert bytes or string
        s = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        video_id, frame_id_str = s.split()
        frame_id = int(frame_id_str)
        vid_frame_map.append((video_id, frame_id))

    return fused_features, vid_frame_map

def compute_dbcv(X, labels, min_cluster_size, metric='euclidean'):
    """
    Compute a Density-Based Clustering Validation (DBCV) score for a clustering
    of data X with labels.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.nan

    # Compute core distances: for each point, distance to its min_cluster_size-th neighbor.
    nbrs = NearestNeighbors(n_neighbors=min_cluster_size, metric=metric).fit(X)
    distances, _ = nbrs.kneighbors(X)
    core_dists = distances[:, -1]  # core distance for each point

    # Process clusters (ignore noise, label == -1)
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    if len(unique_clusters) == 0:
        return np.nan

    cluster_validities = []  # list of (cluster_size, V_c)
    for c in unique_clusters:
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) < 2:
            continue
        Xc = X[cluster_indices]
        core_sub = core_dists[cluster_indices]

        # Compute intra-cluster mutual-reachability distances:
        D_intra = pairwise_distances(Xc, metric=metric)
        M = np.maximum(np.maximum(core_sub[:, None], core_sub[None, :]), D_intra)
        mst = minimum_spanning_tree(M)
        mst = mst.toarray()
        mst_edges = mst[mst > 0]
        intra_connectivity = np.mean(mst_edges) if mst_edges.size > 0 else 0.0

        # Compute separation:
        outside_indices = np.where(labels != c)[0]
        if len(outside_indices) == 0:
            separation = np.max(M)
        else:
            X_out = X[outside_indices]
            D_inter = pairwise_distances(Xc, X_out, metric=metric)
            core_out = core_dists[outside_indices]
            M_inter = np.maximum(np.maximum(core_sub[:, None], core_out[None, :]), D_inter)
            separation = np.min(M_inter)

        if separation <= 0:
            V_c = -1
        else:
            V_c = (separation - intra_connectivity) / separation
        cluster_validities.append((len(cluster_indices), V_c))

    if len(cluster_validities) == 0:
        return np.nan

    total_points = sum(size for size, _ in cluster_validities)
    dbcv_score = sum(size * v for size, v in cluster_validities) / total_points
    return dbcv_score

def grid_search_hdbscan(X, grid_min_cluster_size, grid_min_samples, metric='euclidean'):
    """
    For a given dataset X, perform a grid search over candidate values for
    min_cluster_size and min_samples for HDBSCAN. For each parameter combination,
    compute the clustering and its DBCV score. Return the best DBCV score, the best
    parameter combination, and the corresponding cluster labels.
    """
    best_dbcv = -np.inf
    best_params = None
    best_labels = None
    for mcs in grid_min_cluster_size:
        if len(X) < mcs:
            continue
        for ms in grid_min_samples:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric=metric)
            labels = clusterer.fit_predict(X)
            try:
                score = compute_dbcv(X, labels, mcs, metric=metric)
            except Exception as e:
                score = -np.inf
            if np.isnan(score):
                score = -np.inf
            if score > best_dbcv:
                best_dbcv = score
                best_params = (mcs, ms)
                best_labels = labels
    return best_dbcv, best_params, best_labels

def process_video(video_id, sub_feats, grid_min_cluster_size, grid_min_samples, metric='euclidean'):
    """
    Perform grid search for a single video and return the results.
    """
    if len(sub_feats) < min(grid_min_cluster_size):
        # Not enough frames for clustering.
        best_score = np.nan
        best_params = (None, None)
        best_labels = np.full((len(sub_feats),), -1, dtype=int)
    else:
        best_score, best_params, best_labels = grid_search_hdbscan(sub_feats, grid_min_cluster_size, grid_min_samples, metric)
    return video_id, best_score, best_params, best_labels

def cluster(visualize = False):
    # Set parameters directly without using command-line arguments.
    fused_file = "feature_extraction/cross_modal_features/fused_features.h5"

    # Define grid search parameter ranges.
    grid_min_cluster_size = [3,4,5]
    grid_min_samples = [3,4,5]
    visual_count = 5

    # 1) Load fused features + mapping.
    fused_features, vid_frame_map = load_fused_features(fused_file)
    N = fused_features.shape[0]
    print(f"Loaded {N} fused feature vectors from {fused_file}")

    # 2) Group frames by video_id.
    video_dict = defaultdict(lambda: {"indices": [], "features": None})
    for i, (vid, fid) in enumerate(vid_frame_map):
        video_dict[vid]["indices"].append(i)
    for vid in video_dict.keys():
        idx_list = video_dict[vid]["indices"]
        sub_feats = fused_features[idx_list, :]  # shape (#frames_of_video, d)
        video_dict[vid]["features"] = sub_feats

    del fused_features  # free memory
    gc.collect()

    # 3) Process each video in parallel with a progress bar.
    cluster_dict = {}
    dbcv_dict = {}   # overall best DBCV score for each video.
    best_param_dict = {}  # best parameter combination for each video.
    results = {}
    all_video_ids = list(video_dict.keys())
    print("Performing grid search for optimal HDBSCAN parameters on each video...")

    with ProcessPoolExecutor() as executor:
        # Submit tasks for each video.
        futures = {executor.submit(process_video, video_id, video_dict[video_id]["features"],
                                     grid_min_cluster_size, grid_min_samples, 'euclidean'): video_id
                   for video_id in all_video_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Videos Processed"):
            video_id, best_score, best_params, best_labels = future.result()
            cluster_dict[video_id] = best_labels
            dbcv_dict[video_id] = best_score
            best_param_dict[video_id] = best_params
            num_clusters = len(np.unique(best_labels[best_labels >= 0]))
            #print(f"Video {video_id}: {video_dict[video_id]['features'].shape[0]} frames, found {num_clusters} clusters; "
            #      f"Best DBCV = {best_score:.4f} with min_cluster_size = {best_params[0]} and min_samples = {best_params[1]}.")

    # 4) (Optional) Visualize up to 5 random videos in 2D with UMAP, saving to PDF.
    if visualize:
        random_vids = random.sample(all_video_ids, k=min(visual_count, len(all_video_ids)))
        print(f"Visualizing up to 5 random videos (PDF): {random_vids}")
        for vid in random_vids:
            sub_feats = video_dict[vid]["features"]
            labels = cluster_dict[vid]
            outlier_mask = (labels == -1)
            print(f"Applying UMAP to video {vid} for cluster visualization (2D)...")
            reducer = umap.UMAP(n_components=2)
            embedding = reducer.fit_transform(sub_feats)
            jitter_scale = 0.02 * np.std(embedding)
            embedding_jittered = embedding + np.random.normal(scale=jitter_scale, size=embedding.shape)
            plt.figure(figsize=(12,10))
            plt.scatter(embedding_jittered[outlier_mask, 0],
                        embedding_jittered[outlier_mask, 1],
                        c="lightgray", s=30, alpha=0.7, label="Outliers")
            unique_labels = np.unique(labels[~outlier_mask])
            cmap = plt.get_cmap("viridis", len(unique_labels))
            for i, cl in enumerate(unique_labels):
                idx = (labels == cl)
                plt.scatter(embedding_jittered[idx, 0],
                            embedding_jittered[idx, 1],
                            s=30, alpha=0.7, label=f"Cluster {cl}",
                            c=[cmap(i)])
            plt.title(f"HDBSCAN Clusters (UMAP 2D)\nVideo: {vid}", fontsize=14)
            plt.legend(markerscale=1.5, loc="upper right")
            plt.tight_layout()
            pdf_name = f"clustering/cluster_examples/{vid}_clusters.pdf"
            plt.savefig(pdf_name, bbox_inches='tight')
            plt.close()
            #print(f"Saved clustering plot for video {vid} to {pdf_name}")

    # 5) Save cluster labels and grid search results per video to HDF5.
    output_file = "clustering/clusters_per_video.h5"
    print(f"Saving cluster labels and grid search results to clustering/{output_file} ...")
    with h5py.File(output_file, "w") as f:
        for video_id in all_video_ids:
            grp = f.create_group(video_id)
            idx_list = video_dict[video_id]["indices"]
            labels_array = cluster_dict[video_id]
            frame_ids = [vid_frame_map[idx][1] for idx in idx_list]
            grp.create_dataset("frame_ids", data=frame_ids)
            grp.create_dataset("cluster_labels", data=labels_array)
            grp.attrs['dbcv_score'] = dbcv_dict.get(video_id, np.nan)
            best_params = best_param_dict.get(video_id, (None, None))
            grp.attrs['best_min_cluster_size'] = best_params[0] if best_params[0] is not None else -1
            grp.attrs['best_min_samples'] = best_params[1] if best_params[1] is not None else -1

if __name__ == "__main__":
    cluster()
