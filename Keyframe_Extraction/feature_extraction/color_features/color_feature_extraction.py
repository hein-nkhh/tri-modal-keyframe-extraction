import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import gc
import h5py

def get_videos(directory='../../test_videos'):
    """
    Gets the videos from the 'videos' directory and returns the video file paths as a list.
    """
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Get list of all video files in the directory
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"No video files found in '{directory}'. Please add videos to process.")

    return video_files

def calculate_color_histogram(frame_lab, bins=256):
    """
    Calculates the color histogram for a frame in Lab color space.

    Returns:
    - hist_feature: concatenated histogram of L, a, b channels.
    """
    # Calculate histogram for each channel
    l_hist = cv2.calcHist([frame_lab], [0], None, [bins], [0, 256])
    a_hist = cv2.calcHist([frame_lab], [1], None, [bins], [0, 256])
    b_hist = cv2.calcHist([frame_lab], [2], None, [bins], [0, 256])

    # Normalize histograms
    l_hist = l_hist / np.sum(l_hist)
    a_hist = a_hist / np.sum(a_hist)
    b_hist = b_hist / np.sum(b_hist)

    # Concatenate histograms into a single feature vector
    hist_feature = np.concatenate((l_hist.flatten(), a_hist.flatten(), b_hist.flatten()))

    return hist_feature

def calculate_color_moments(frame_lab):
    """
    Calculates the color moments (mean, variance, skewness) for each channel in Lab color space.

    Returns:
    - moments_feature: concatenated moments for L, a, b channels.
    """
    moments = []
    for i in range(3):  # L, a, b channels
        channel = frame_lab[:, :, i].flatten()
        mean = np.mean(channel)
        variance = np.var(channel)
        skewness = np.mean((channel - mean) ** 3) / (np.std(channel) ** 3 + 1e-8)
        moments.extend([mean, variance, skewness])
    moments_feature = np.array(moments)
    return moments_feature

def calculate_colorfulness_metric(frame_lab):
    """
    Calculates the colorfulness metric of the frame.

    Returns:
    - colorfulness: scalar value representing the colorfulness of the frame.
    """
    # Split channels
    a = frame_lab[:, :, 1]
    b = frame_lab[:, :, 2]

    # Compute the mean and standard deviation of each channel
    std_a, std_b = np.std(a), np.std(b)
    mean_a, mean_b = np.mean(a), np.mean(b)

    # Compute the colorfulness metric
    std_root = np.sqrt(std_a ** 2 + std_b ** 2)
    mean_root = np.sqrt(mean_a ** 2 + mean_b ** 2)
    colorfulness = std_root + (0.3 * mean_root)
    return np.array([colorfulness])

def process_frame(frame):
    """
    Processes a frame: converts to Lab color space and calculates color features.

    Returns:
    - features: concatenated feature vector including histograms, moments, and colorfulness metric.
    """
    # Convert BGR to Lab directly using OpenCV
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Calculate color features
    hist_feature = calculate_color_histogram(frame_lab, bins=256)
    moments_feature = calculate_color_moments(frame_lab)
    colorfulness_feature = calculate_colorfulness_metric(frame_lab)

    # Concatenate all features into a single feature vector
    features = np.concatenate((hist_feature, moments_feature, colorfulness_feature))

    return features

def process_video(video_file, batch_size, frame_interval):
    """
    Processes a single video: reads frames and computes color features.

    Returns:
    - (video_name, features_list): Tuple containing the video name and list of tuples (frame_id, feature_vector).
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features_list = []

    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", leave=False)

    # Define ThreadPoolExecutor for per-frame processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        frames = []
        frame_ids = []
        futures = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frames.append(frame)
                frame_ids.append(frame_idx)  # Keep track of the frame number

                if len(frames) == batch_size:
                    # Process batch of frames in parallel
                    futures = [executor.submit(process_frame, f) for f in frames]
                    for idx, future in enumerate(futures):
                        feature_vector = future.result()
                        features_list.append((frame_ids[idx], feature_vector))
                    frames = []
                    frame_ids = []
                    # Free up memory
                    gc.collect()
            frame_idx += 1
            pbar.update(1)

        # Process any remaining frames
        if frames:
            futures = [executor.submit(process_frame, f) for f in frames]
            for idx, future in enumerate(futures):
                feature_vector = future.result()
                features_list.append((frame_ids[idx], feature_vector))
            frames = []
            frame_ids = []
            gc.collect()

    pbar.close()
    cap.release()

    # Return the video name and features_list
    return (video_name, features_list)

def process_video_wrapper(args):
    """
    Wrapper function for process_video to enable passing multiple arguments to Pool.imap_unordered.
    """
    return process_video(*args)

def read_frames_parallel(video_files, batch_size, frame_interval):
    """
    Reads videos in parallel, extracts frames, and computes color features.

    Returns:
    - features_dict: Dictionary with video names as keys and lists of tuples (frame_id, feature_vector) as values.
    """
    features_dict = {}

    # Define the number of processes for multiprocessing
    num_processes = 10

    with Pool(processes=num_processes) as pool:
        # Prepare arguments for each video
        args_list = [(vf, batch_size, frame_interval) for vf in video_files]
        # Process videos in parallel
        for result in tqdm(pool.imap_unordered(process_video_wrapper, args_list),
                           total=len(video_files), desc="Processing Videos"):
            video_name, features_list = result
            features_dict[video_name] = features_list
            # Free up memory
            del features_list
            gc.collect()

    return features_dict

def save_features_dict_hdf5(features_dict, filename):
    """
    Saves the features_dict to an HDF5 file for future use.

    Parameters:
    - features_dict: Dictionary with video names as keys and lists of tuples (frame_id, feature_vector) as values.
    - filename: Name of the HDF5 file to save the features_dict.
    """
    with h5py.File(filename, 'w') as h5f:
        for video_name, features_list in features_dict.items():
            # Prepare data and frame_ids
            frame_ids = [item[0] for item in features_list]
            features_array = np.array([item[1] for item in features_list])
            # Create a group for each video
            grp = h5f.create_group(video_name)
            # Save frame_ids and features
            grp.create_dataset('frame_ids', data=frame_ids)
            grp.create_dataset('features', data=features_array)
    print(f"Features dictionary saved to {filename}")


def extract_color_features(video_folder='../../test_videos', frame_interval = 10):
    # Step 1: Get list of video files
    video_files = get_videos(video_folder)
    if not video_files:
        return

    # Step 2: Read frames, compute features in parallel
    features_dict = read_frames_parallel(video_files, batch_size=32, frame_interval=frame_interval)

    # Step 3: Save the features_dict to an HDF5 file
    save_features_dict_hdf5(features_dict, filename='feature_extraction/color_features/color_features_dict.h5')

if __name__ == "__main__":
    extract_color_features()
