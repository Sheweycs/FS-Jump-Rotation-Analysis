#!/usr/bin/env python3
"""
Skeleton Video Extraction
Extract skeleton overlay videos from original videos using MediaPipe Pose.

Input: Original video files (.mp4, .avi, .mov)
Output: Videos with skeleton overlay (*_SK.mp4)
"""
import os
import cv2
import mediapipe as mp


class SkeletonVideoExtractor:
    """Extract skeleton videos using MediaPipe Pose."""

    LANDMARK_SPEC = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 0, 200), thickness=6, circle_radius=5  # Dark red (BGR)
    )
    CONNECTION_SPEC = mp.solutions.drawing_utils.DrawingSpec(
        color=(200, 0, 0), thickness=4  # Dark blue (BGR)
    )

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )

    def _is_reliable(self, landmarks, width, height):
        """Check if pose detection is reliable."""
        if not landmarks:
            return False
        key_points = [0, 11, 12, 23, 24]  # nose, shoulders, hips
        reliable = sum(1 for idx in key_points if landmarks.landmark[idx].visibility > 0.7)
        if reliable < 4:
            return False
        nose = landmarks.landmark[0]
        margin = 0.1
        if not (width * margin < nose.x * width < width * (1 - margin) and
                height * margin < nose.y * height < height * (1 - margin)):
            return False
        return True

    def process_video(self, input_path, output_path):
        """Process a video and output skeleton overlay video."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated = frame.copy()

            if results.pose_landmarks and self._is_reliable(results.pose_landmarks, width, height):
                self.mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.LANDMARK_SPEC, self.CONNECTION_SPEC
                )

            out.write(annotated)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Progress: {frame_idx}/{total} ({frame_idx/total*100:.1f}%)")

        cap.release()
        out.release()
        print(f"  Saved: {output_path}")
        return True

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


def batch_process(input_dir, output_dir):
    """Batch process all videos in subdirectories."""
    video_ext = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']

    # Find subdirectories
    subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')]

    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return

    # Create output subdirectories
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, os.path.basename(subdir)), exist_ok=True)

    # Collect videos
    videos = []
    for subdir in subdirs:
        subdir_name = os.path.basename(subdir)
        for f in os.listdir(subdir):
            if any(f.endswith(ext) for ext in video_ext):
                videos.append({'path': os.path.join(subdir, f), 'name': f, 'subdir': subdir_name})

    print(f"Found {len(videos)} videos in {len(subdirs)} subdirectories")

    extractor = SkeletonVideoExtractor()
    success = 0

    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {v['subdir']}/{v['name']}")
        name_no_ext = os.path.splitext(v['name'])[0]
        output_path = os.path.join(output_dir, v['subdir'], f"{name_no_ext}_SK.mp4")
        if extractor.process_video(v['path'], output_path):
            success += 1

    print(f"\nComplete: {success}/{len(videos)} videos processed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract skeleton videos from original videos")
    parser.add_argument('--input', type=str, default='.', help='Input directory with video subdirectories')
    parser.add_argument('--output', type=str, default='skeleton_output', help='Output directory')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist")
        exit(1)

    batch_process(input_dir, output_dir)
