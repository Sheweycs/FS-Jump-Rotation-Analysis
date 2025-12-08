"""
Result Visualization for Figure Skating Jump Classification
Overlays prediction results on skeleton videos.

Input:
    - results_file: CSV/JSON file with columns [video_key, true_label, pred_label, confidence]
    - skeleton videos in base_dir/{folder}/*_SK.mp4

Output:
    - Videos with prediction overlay in output_dir/
"""

import os
import glob
import argparse
import csv
import json
import cv2


CLASS_NAMES = {
    0: "Full Rotation",
    1: "Under-rotated"
}

COLORS = {
    "success": (0, 200, 0),
    "failure": (0, 0, 200),
    "text_bg": (0, 0, 0),
    "white": (255, 255, 255),
}

FOLDER_MAP = {
    0: "0_full rotation",
    1: "1_on the quarter",  # or 2_underrotated, 3_downgraded
}


# Step 1: Load Prediction Results

def load_results_csv(csv_path):
    """
    Load prediction results from CSV file.
    Expected columns: video_key, true_label, pred_label, confidence (optional)

    Returns:
        dict: {video_key: {"true_label": int, "pred_label": int, "confidence": float}}
    """
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_key = row["video_key"]
            results[video_key] = {
                "true_label": int(row["true_label"]),
                "pred_label": int(row["pred_label"]),
                "confidence": float(row.get("confidence", 1.0)),
            }
    return results


def load_results_json(json_path):
    """
    Load prediction results from JSON file.
    Expected format: {video_key: {"true_label": int, "pred_label": int, "confidence": float}}
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def load_results(file_path):
    """Load results from CSV or JSON file."""
    if file_path.endswith('.csv'):
        return load_results_csv(file_path)
    elif file_path.endswith('.json'):
        return load_results_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")


# Step 2: Draw Overlay on Video Frames

def get_label_name(label_idx):
    return CLASS_NAMES.get(label_idx, f"Unknown ({label_idx})")


def draw_prediction_overlay(frame, true_label, pred_label, video_name):
    """Draw prediction overlay on a single frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    is_correct = (true_label == pred_label)
    box_color = COLORS["success"] if is_correct else COLORS["failure"]

    # Box dimensions (bottom-right corner)
    box_width, box_height, margin = 500, 220, 15
    box_x = w - box_width - margin
    box_y = h - box_height - margin

    # Draw semi-transparent background
    cv2.rectangle(overlay, (box_x, box_y), (w - margin, h - margin), COLORS["text_bg"], -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # Text settings
    font = cv2.FONT_HERSHEY_DUPLEX
    text_x = box_x + 25
    y = box_y + 50

    # Line 1: CORRECT/WRONG
    cv2.putText(frame, "CORRECT" if is_correct else "WRONG", (text_x, y),
                font, 1.3, box_color, 2)

    # Line 2: Video name
    y += 55
    cv2.putText(frame, video_name[:40], (text_x, y), font, 0.8, COLORS["white"], 1)

    # Line 3: True label
    y += 55
    cv2.putText(frame, f"True: {get_label_name(true_label)}", (text_x, y),
                font, 1.3, COLORS["white"], 2)

    # Line 4: Predicted label
    y += 55
    cv2.putText(frame, f"Pred: {get_label_name(pred_label)}", (text_x, y),
                font, 1.3, COLORS["white"], 2)

    return frame


# Step 3: Process Video and Generate Output

def process_video(video_path, true_label, pred_label, output_path):
    """Read video, add overlay to each frame, save output."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    video_name = os.path.basename(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_prediction_overlay(frame, true_label, pred_label, video_name)
        out.write(frame)
        count += 1

    cap.release()
    out.release()
    print(f"  Saved: {output_path} ({count} frames)")
    return True


def find_video(base_dir, video_key, true_label):
    """Find skeleton video file for a given video_key."""
    # Try the folder corresponding to true_label
    folder = FOLDER_MAP.get(true_label, "0_full rotation")
    video_path = os.path.join(base_dir, folder, f"{video_key}.mp4")
    if os.path.exists(video_path):
        return video_path

    # Search all folders
    for folder in ["0_full rotation", "1_on the quarter", "2_underrotated", "3_downgraded"]:
        video_path = os.path.join(base_dir, folder, f"{video_key}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def generate_result_videos(results, base_dir, output_dir):
    """Generate visualization videos for all results."""
    os.makedirs(output_dir, exist_ok=True)

    total, success, correct = 0, 0, 0

    for video_key, info in results.items():
        video_path = find_video(base_dir, video_key, info["true_label"])
        if not video_path:
            print(f"  Skip: {video_key} (video not found)")
            continue

        total += 1
        is_correct = info["true_label"] == info["pred_label"]
        if is_correct:
            correct += 1

        tag = "correct" if is_correct else "wrong"
        output_path = os.path.join(output_dir, f"{tag}_{os.path.basename(video_path)}")

        print(f"Processing: {video_key}")
        if process_video(video_path, info["true_label"], info["pred_label"], output_path):
            success += 1

    print(f"\nComplete! {success}/{total} videos generated")
    print(f"Accuracy: {correct}/{total} ({100*correct/max(1,total):.1f}%)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result visualization videos")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results file (CSV or JSON)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Directory containing skeleton videos")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    BASE_DIR = args.base_dir or os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = args.output_dir or os.path.join(BASE_DIR, "result_videos")

    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} results")

    generate_result_videos(results, BASE_DIR, OUTPUT_DIR)
