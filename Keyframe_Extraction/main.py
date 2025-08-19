from feature_extraction.color_features import color_feature_extraction
from feature_extraction.image_features import image_feature_extraction
from feature_extraction.content_features import content_feature_extraction
from feature_extraction.cross_modal_features import feature_level_fusion
from clustering import cluster_frames
from frame_selection import candidate_frame_selection
import time

def main():

    start = time.time()

    video_folder = "evaluation/"
    dataset_name = "Summe"

    full_path = video_folder + dataset_name

    print("Feature Extraction is started...")

    print("Extracting Color Features...")
    color_feature_extraction.extract_color_features(full_path, frame_interval=1)

    print("Extracting Image Features...")
    image_feature_extraction.extract_image_features(full_path, frame_interval=1)

    print("Extracting Content Features...")
    content_feature_extraction.extract_content_features(full_path, frame_interval=1)

    print("Fusing the Features...")
    feature_level_fusion.fuse_features()

    print("Clustering the Fused Features...")
    cluster_frames.cluster(visualize=False)

    print("Selecting the Candidate Keyframes...")
    candidate_frame_selection.find_candidate_keyframes(full_path)

    end = time.time()
    print("Time passed:" + str(end-start))
if __name__ == '__main__':
    main()

