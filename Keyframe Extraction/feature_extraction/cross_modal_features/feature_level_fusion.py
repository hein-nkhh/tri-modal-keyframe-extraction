import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gc
import h5py

def read_h5_by_video(h5_path):
    """
    Reads an HDF5 file where each top-level group corresponds to a distinct 'video_id'.
    Within each 'video_id' group, we expect two datasets:
      1) 'frame_ids': 1D array of shape (N,) listing the frames or frame indices
      2) 'features': 2D array of shape (N, D), one feature vector per frame

    Parameters
    ----------
    h5_path : str
        File path to the HDF5 file.

    Returns
    -------
    video_dict : dict
        A dictionary where each key is a video_id (str), and each value is a tuple:
        (frame_ids_array, features_array).
        - frame_ids_array has shape (N,).
        - features_array has shape (N, D).
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")

    video_dict = {}
    with h5py.File(h5_path, 'r') as f:
        # Iterate over each top-level group, which should be a video_id
        for video_id in f.keys():
            grp = f[video_id]

            # Validate the existence of 'frame_ids' and 'features'
            if 'frame_ids' not in grp or 'features' not in grp:
                print(f"Warning: group '{video_id}' missing 'frame_ids' or 'features'. Skipping.")
                continue

            # Read datasets
            frame_ids = grp['frame_ids'][:]
            features = grp['features'][:]

            # Store them in the dictionary for this video_id
            video_dict[video_id] = (frame_ids, features)

    return video_dict


def fuse_features():
    # -------------------------------------------------------------------------
    # 1) Define paths to the HDF5 files for color, image, and content features
    # -------------------------------------------------------------------------
    pca_dim = 512

    color_h5 = "feature_extraction/color_features/color_features_dict.h5"
    image_h5 = "feature_extraction/image_features/image_features_dict.h5"
    content_h5 = "feature_extraction/content_features/content_description_features.h5"

    # -------------------------------------------------------------------------
    # 2) Read each HDF5 into a dictionary where each key is a video_id,
    #    and each value is (frame_ids, features).
    # -------------------------------------------------------------------------
    color_data = read_h5_by_video(color_h5)
    image_data = read_h5_by_video(image_h5)
    content_data = read_h5_by_video(content_h5)

    # Lists to accumulate feature arrays across all videos
    color_list = []
    image_list = []
    content_list = []
    vid_frame_map = []  # Store (video_id, frame_id) for each row in final arrays

    # -------------------------------------------------------------------------
    # 3) Merge features for each video_id. We assume all files share identical
    #    video_ids and frame ordering, so no intersection or re-indexing is needed.
    # -------------------------------------------------------------------------
    for video_id in color_data.keys():
        # Basic check to ensure other files have this video_id
        if video_id not in image_data or video_id not in content_data:
            raise ValueError(f"video_id '{video_id}' not found in one of the HDF5 files.")

        # Unpack the arrays for color, image, content features for this video
        color_frame_ids, color_feats = color_data[video_id]
        image_frame_ids, image_feats = image_data[video_id]
        content_frame_ids, content_feats = content_data[video_id]

        # Confirm that each block has the same number of frames
        N_c = len(color_frame_ids)
        N_i = len(image_frame_ids)
        N_t = len(content_frame_ids)
        if not (N_c == N_i == N_t):
            raise ValueError(f"Mismatch in frame counts for video {video_id}: "
                             f"color={N_c}, image={N_i}, content={N_t}")

        # Append features to global lists
        color_list.append(color_feats)  # (N, color_dim)
        image_list.append(image_feats)  # (N, image_dim)
        content_list.append(content_feats)  # (N, text_dim)

        # Store (video_id, frame_id) for each row
        for fid in color_frame_ids:
            vid_frame_map.append((video_id, fid))

        # If memory usage is a concern, consider partial writes or chunk processing

    # -------------------------------------------------------------------------
    # 4) Convert each accumulated list into a single ndarray by stacking
    # -------------------------------------------------------------------------
    color_arr = np.vstack(color_list)  # (total_frames, color_dim)
    image_arr = np.vstack(image_list)  # (total_frames, image_dim)
    content_arr = np.vstack(content_list)  # (total_frames, text_dim)

    print(f"Global color shape:   {color_arr.shape}")
    print(f"Global image shape:   {image_arr.shape}")
    print(f"Global content shape: {content_arr.shape}")

    total_frames = color_arr.shape[0]

    # -------------------------------------------------------------------------
    # 5) Normalize each block using StandardScaler
    # -------------------------------------------------------------------------
    color_scaler = StandardScaler()
    image_scaler = StandardScaler()
    content_scaler = StandardScaler()

    color_norm = color_scaler.fit_transform(color_arr)
    image_norm = image_scaler.fit_transform(image_arr)
    content_norm = content_scaler.fit_transform(content_arr)

    # -------------------------------------------------------------------------
    # 6) Concatenate normalized blocks into one fused feature vector per frame
    # -------------------------------------------------------------------------
    fused = np.concatenate([color_norm, image_norm, content_norm], axis=1)
    print(f"Fused feature shape before PCA: {fused.shape}")

    # -------------------------------------------------------------------------
    # 7) Apply PCA to reduce dimensionality of the fused features
    # -------------------------------------------------------------------------
    print(f"Applying PCA to reduce dimension to {pca_dim} ...")
    pca = PCA(n_components=pca_dim)
    fused_pca = pca.fit_transform(fused)
    print(f"Fused shape after PCA: {fused_pca.shape}")

    # -------------------------------------------------------------------------
    # 8) Save the fused features and the (video_id, frame_id) mapping
    # -------------------------------------------------------------------------
    output_h5 = "feature_extraction/cross_modal_features/fused_features.h5"
    print(f"Saving fused features to feature_extraction/cross_modal_features/{output_h5} ...")


    with h5py.File(output_h5, "w") as f:
        # Save the fused PCA features
        f.create_dataset("fused_features", data=fused_pca)

        # Save the (video_id, frame_id) map as a string array
        dt = h5py.special_dtype(vlen=str)
        str_ids = [f"{vf[0]} {vf[1]}" for vf in vid_frame_map]
        f.create_dataset("vid_frame_map", (total_frames,), dtype=dt)
        f["vid_frame_map"][:] = str_ids

    # Optional cleanup to free memory
    del color_arr, image_arr, content_arr
    gc.collect()


if __name__ == "__main__":
    fuse_features()
