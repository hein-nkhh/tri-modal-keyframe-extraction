import os
import cv2
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

def quality_test(frame):
    # Convert to grayscale for intensity and variance checks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    var_val = np.var(gray)

    # Edge density using Canny
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # Blurriness detection using the variance of the Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Solid color detection using color histogram variance
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
    color_var = np.var(np.concatenate([hist_b, hist_g, hist_r]))

    # Approximate saliency using intensity variance in regions
    h, w = gray.shape
    center_region = gray[h//4:h*3//4, w//4:w*3//4]
    saliency_score = np.var(center_region) / (np.var(gray) + 1e-5)

    # Text detection using simple MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    text_region_count = len(regions)

    # Keypoint detection using ORB
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    keypoint_count = len(keypoints)

    # Thresholds
    if mean_val < 30 or mean_val > 220:  # Too dark or too bright
        if saliency_score < 0.2 and text_region_count < 5:
            return False

    if var_val < 50:  # Very low contrast
        if keypoint_count < 10 and text_region_count < 5:
            return False

    if edge_density < 0.005:  # Almost no edges
        if keypoint_count < 10 and saliency_score < 0.2:
            return False

    if laplacian_var < 20:  # Blurry image
        if keypoint_count < 10:
            return False

    if color_var < 10:  # Solid color frame
        if text_region_count < 5 and saliency_score < 0.2:
            return False

    return True

def duplicate_check(new_frame, accepted_frames, threshold):
    """
    Checks if the new_frame is a duplicate of any frame in accepted_frames using SSIM.
    Returns True if any SSIM value is above the threshold, otherwise False.
    """
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    for frame in accepted_frames:
        ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sim, _ = ssim(new_gray, ref_gray, full=True)
        if sim >= threshold:
            return True
    return False


# ---------------------------
# Frame Extraction Function
# ---------------------------
def extract_candidate_frame(video_path, candidate_frame_id, output_folder, accepted_frames, dup_threshold):
    """
    Extract a specific frame from the video and apply a quality test and duplicate check.
    If the frame passes both tests, save it as a PNG image.

    Parameters:
      video_path : str - path to the video file.
      candidate_frame_id : int - frame index to extract.
      output_folder : str - directory in which to save the PNG.
      accepted_frames : list - list of previously accepted frames for duplicate checking.
      dup_threshold : float - SSIM threshold for duplicate detection.

    Returns:
      saved_file : str or None - path to the saved PNG if successful; else None.
      frame : np.ndarray or None - the extracted frame if saved; else None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"Error: Could not extract frame {candidate_frame_id} from {video_path}")
        return None, None

    if not quality_test(frame):
        return None, None

    if duplicate_check(frame, accepted_frames, threshold=dup_threshold):
        return None, None

    os.makedirs(output_folder, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_folder, f"{video_name}_frame{candidate_frame_id}.png")
    cv2.imwrite(output_file, frame)
    return output_file, frame


# ---------------------------
# Video Candidate Processing Function
# ---------------------------
def process_video_candidates(video_id, group_df, videos_dir, output_folder, dup_threshold):
    """
    Processes candidate keyframes for a single video.
    For each candidate frame (as specified in group_df), extracts the frame
    from the video, applies quality and duplicate tests, and saves the frame as PNG.
    Returns a summary for the video.

    Parameters:
      video_id : str - the video identifier.
      group_df : DataFrame - rows from the candidate_keyframes CSV for this video.
      videos_dir : str - directory containing video files (assumed as video_id.mp4).
      output_folder : str - folder to save extracted PNG images.
      dup_threshold : float - SSIM threshold for duplicate detection.

    Returns:
      A tuple: (video_id, processed_count, dropped_count, saved_files)
    """
    processed_count = 0
    dropped_count = 0
    saved_files = []
    accepted_frames = []  # List of accepted frames (BGR) for duplicate checking

    video_file = os.path.join(videos_dir, f"{video_id}.mp4")
    if not os.path.isfile(video_file):
        print(f"Video file for {video_id} not found. Skipping.")
        return video_id, 0, 0, []

    for _, row in group_df.iterrows():
        candidate_frame_id = int(row['candidate_frame_id'])
        processed_count += 1
        saved_file, frame = extract_candidate_frame(video_file, candidate_frame_id, output_folder,
                                                    accepted_frames, dup_threshold)
        if saved_file is None:
            dropped_count += 1
        else:
            saved_files.append(saved_file)
            accepted_frames.append(frame)
    return video_id, processed_count, dropped_count, saved_files


# ---------------------------
# Main Execution Function
# ---------------------------
def find_keyframes(videos_dir='../../test_videos'):
    # File paths and directories.
    csv_file = "frame_selection/candidate_keyframes.csv"  # CSV with columns: video_id, candidate_frame_id
    output_folder = "key_frames/" + videos_dir  # Folder to save candidate frames as PNG

    dup_threshold = 0.8

    # Read candidate keyframes CSV.
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} candidate keyframes from {csv_file}")

    # Group by video_id.
    grouped = df.groupby("video_id")

    # Prepare a list of tasks: one per video.
    tasks = [(video_id, group.copy(), videos_dir, output_folder) for video_id, group in grouped]

    results_summary = {}  # video_id: (processed, dropped, saved_files)

    # Process each video in parallel.
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_video_candidates, video_id, group_df, videos_dir, output_folder, dup_threshold = dup_threshold): video_id
                   for video_id, group_df, videos_dir, output_folder in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Videos Processed"):
            video_id, proc, drop, saved = future.result()
            results_summary[video_id] = (proc, drop, saved)
            print(f"Video {video_id}: Processed {proc} candidates, dropped {drop}, saved {len(saved)} frames.")

    total_saved = sum(len(saved) for (_, _, saved) in results_summary.values())
    print(f"Total keyframes saved: {total_saved}")
    gc.collect()


if __name__ == "__main__":
    find_keyframes()
