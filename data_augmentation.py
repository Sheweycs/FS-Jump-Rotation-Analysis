import os
import cv2

# Folder paths
folders = {
    'full': '0_full rotation',
    'quarter': '1_on the quarter',
    'under': '2_underrotated',
    'down': '3_downgraded'
}

TARGET_COUNT = 500

def count_videos(folder_path):
    """Count video files in a folder"""
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    return len([f for f in os.listdir(folder_path)
                if any(f.endswith(ext) for ext in video_extensions)])

def get_video_files(folder_path):
    """Get all video files in a folder"""
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    return [f for f in os.listdir(folder_path)
            if any(f.endswith(ext) for ext in video_extensions)]

def flip_video(input_path, output_path):
    """Horizontally flip a video"""
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped_frame = cv2.flip(frame, 1)
        out.write(flipped_frame)

    cap.release()
    out.release()
    print(f"Created flipped video: {output_path}")

def rotate_video(input_path, output_path, angle):
    """Rotate a video by a given angle"""
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
        out.write(rotated_frame)

    cap.release()
    out.release()
    print(f"Created rotated video ({angle}deg): {output_path}")

def augment_group(folder_list, group_name):
    """Augment videos in a group of folders until reaching TARGET_COUNT"""
    # Count current videos
    current_count = sum(count_videos(f) for f in folder_list)
    print(f"\n=== {group_name} ===")
    print(f"Current count: {current_count}")

    if current_count >= TARGET_COUNT:
        print(f"Already at {TARGET_COUNT}, no augmentation needed.")
        return

    # Collect all original videos (not augmented)
    original_videos = []
    for folder in folder_list:
        for video in get_video_files(folder):
            if '_flipped' not in video and '_rot' not in video:
                original_videos.append((folder, video))

    # Step 1: Flip all original videos
    print(f"\nStep 1: Flipping videos...")
    for folder, video in original_videos:
        if current_count >= TARGET_COUNT:
            print(f"Reached {TARGET_COUNT}, stopping.")
            return

        name, ext = os.path.splitext(video)
        output_path = os.path.join(folder, f"{name}_flipped{ext}")

        if os.path.exists(output_path):
            continue

        try:
            flip_video(os.path.join(folder, video), output_path)
            current_count += 1
        except Exception as e:
            print(f"Error processing {video}: {e}")

    if current_count >= TARGET_COUNT:
        print(f"Reached {TARGET_COUNT} after flipping.")
        return

    # Step 2: Rotate original videos
    print(f"\nStep 2: Rotating videos...")
    angle = 10
    for folder, video in original_videos:
        if current_count >= TARGET_COUNT:
            print(f"Reached {TARGET_COUNT}, stopping.")
            return

        name, ext = os.path.splitext(video)
        output_path = os.path.join(folder, f"{name}_rot{angle}{ext}")

        if os.path.exists(output_path):
            continue

        try:
            rotate_video(os.path.join(folder, video), output_path, angle)
            current_count += 1
        except Exception as e:
            print(f"Error processing {video}: {e}")

    print(f"Final count: {current_count}")

def main():
    # Group 1: Full rotation folder
    augment_group([folders['full']], "Full Rotation")

    # Group 2: Other three folders combined
    other_folders = [folders['quarter'], folders['under'], folders['down']]
    augment_group(other_folders, "Other Three Folders")

    print("\n=== Augmentation Complete ===")

if __name__ == "__main__":
    main()
