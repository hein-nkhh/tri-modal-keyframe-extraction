import copy
import cv2
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

def evaluation(keyframe_center, test_index, video_path):
    """
    Evaluates the similarity between keyframes and predicted frames using color histograms.
    
    Args:
        keyframe_center (list): List of actual keyframe indices.
        test_index (list): List of predicted keyframe indices.
        video_path (str): Path to the video file.
    
    Returns:
        float: F-score measuring the matching accuracy.
    """
    def color_histogram(img):
        """
        Computes the color histogram of an image.
        
        Args:
            img (numpy.ndarray): Input image.
        
        Returns:
            numpy.ndarray: Flattened color histogram.
        """
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()
    
    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Compute color histograms for all frames
    features = [color_histogram(frame) for frame in frames]
    
    # Initialize similarity computation
    keyframe_center_copy = copy.deepcopy(keyframe_center)
    test_index_copy = copy.deepcopy(test_index)
    lens_key = len(keyframe_center)
    lens_text = len(test_index)
    simis = []
    
    for i in range(lens_key):
        base = features[keyframe_center_copy[i]]
        cv2.normalize(base, base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        simi = []
        for j in range(lens_text):
            lat = features[test_index_copy[j]]
            cv2.normalize(lat, lat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            similarity = np.dot(base, lat) / (np.linalg.norm(base) * np.linalg.norm(lat))
            simi.append(similarity)
        simis.append(simi)
    
    # Match keyframes with predicted frames based on similarity scores
    matchs = []
    match_index = []
    while lens_key > 0 and lens_text > 0:
        max_num = float('-inf')  # Initialize max similarity
        max_index = None
        
        # Find the best match
        for num_i, row in enumerate(simis):
            for num_j, num in enumerate(row):
                if num > max_num:
                    max_num = num
                    max_index = (num_i, num_j)
        
        # Assign match if similarity is above threshold
        if max_num > 0.9:
            i, j = max_index
            match = (keyframe_center[i], test_index[j])
            match_index.append(test_index[j])
            matchs.append(match)
        
        # Invalidate matched indices
        for row in simis:
            row[j] = -1
        simis[i] = [-1] * len(simis[i])
        lens_key -= 1
        lens_text -= 1
    
    # Compute precision, recall, and F-score
    x_num = len(matchs)
    precision = x_num / len(test_index) if test_index else 0
    recall = x_num / len(keyframe_center) if keyframe_center else 0
    f_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Matched frames:", matchs)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_value:.4f}")
    
    return f_value

# File path configurations
prediction_file = 'prediction_excel_file.xlsx'
video_dir = "videos_dir/"

df = pd.read_excel(prediction_file)

# Extract required columns
video_name = list(df['video_name'])
actual = list(df['actual_frame_number'])
predicted = list(df['predicted_frame_number'])

# Initialize results dictionary
result_dict = {'f_value': []}

# Iterate over videos and evaluate
for i in tqdm(range(len(video_name))): 
    try:
        keyframe_center = ast.literal_eval(actual[i])  # Convert string list to actual list
        test_index = ast.literal_eval(predicted[i])  # Convert string list to actual list
        video_path = video_dir + video_name[i] + ".mp4"
        f_value = evaluation(keyframe_center, test_index, video_path)
        result_dict['f_value'].append(f_value)
    except Exception as e:
        print(f"Error processing video {video_name[i]}: {e}")
        result_dict['f_value'].append(None)  # Append None in case of errors

# Save results to DataFrame
df_result = pd.DataFrame(result_dict)
print(df_result['f_value'].mean())
