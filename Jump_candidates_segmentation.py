'''
Recommend running locally

Using Physics-Based Heuristics (Body Tightness and Angular Velocity) for Jump Candidate segmentations
The following parameters can be adjusted as needed: 
compact_top_ratio=0.2, 
rot_top_ratio=0.1, 
max_clip_sec=5.0, 
min_gap_sec=0.2,
extend_sec=1.0,
min_segment_sec=0.5
'''

import os
import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# 1. Build pose cache for full routine (MediaPipe once, ~15 fps)
# ============================================================
def build_pose_cache(
    video_path: str,
    cache_path: str,
    cache_fps: float = 15.0,
    resize_width: int = 960
):
    """
    Run MediaPipe Pose ONCE on the full routine video at ~cache_fps,
    and save all keypoints into a compressed npz file.

    The cache will contain:
        keypoints: (T, 33, 4) array, [x, y, z, visibility], normalized [0,1]
        times:     (T,) time in seconds from start of video
        frame_idx: (T,) original frame index in the video
        metadata:  dict with video info and cache settings
    """
    if os.path.exists(cache_path):
        print(f"[Cache] Found existing cache: {cache_path}")
        return cache_path

    print("\n" + "=" * 70)
    print("BUILDING POSE CACHE (full routine)")
    print("=" * 70)
    print(f"Video:      {video_path}")
    print(f"Cache path: {cache_path}")
    print(f"Cache FPS:  {cache_fps}")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Step between sampled frames in original frame index
    step = max(1, int(round(orig_fps / cache_fps)))

    print(f"[Info] Original FPS: {orig_fps:.3f}")
    print(f"[Info] Total frames: {total_frames}")
    print(f"[Info] Sampling every {step} frames (~{orig_fps/step:.2f} FPS)")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )

    keypoints_list = []
    times = []
    frame_indices = []

    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames on the sampling grid
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # Optional: resize to speed up Pose inference
        if resize_width is not None and orig_width > resize_width:
            scale = resize_width / orig_width
            frame_resized = cv2.resize(
                frame, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            frame_resized = frame

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Store normalized coordinates (x, y, z, visibility) for 33 joints
            kp = np.zeros((33, 4), dtype=np.float32)
            for i, landmark in enumerate(lm):
                kp[i, 0] = landmark.x
                kp[i, 1] = landmark.y
                kp[i, 2] = landmark.z
                kp[i, 3] = landmark.visibility
        else:
            # No pose detected: store zeros
            kp = np.zeros((33, 4), dtype=np.float32)

        keypoints_list.append(kp)
        times.append(frame_idx / orig_fps)
        frame_indices.append(frame_idx)

        sampled += 1
        frame_idx += 1

        if sampled % 50 == 0:
            print(f"  Processed {sampled} sampled frames...")

    cap.release()
    pose.close()

    keypoints = np.stack(keypoints_list, axis=0) if keypoints_list else np.zeros((0, 33, 4), dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    frame_indices = np.asarray(frame_indices, dtype=np.int32)

    metadata = {
        "video_name": os.path.basename(video_path),
        "video_path": os.path.abspath(video_path),
        "orig_fps": float(orig_fps),
        "total_frames": int(total_frames),
        "orig_width": int(orig_width),
        "orig_height": int(orig_height),
        "cache_fps": float(orig_fps / step),
        "step": int(step),
        "num_samples": int(len(times)),
        "keypoint_format": "(T, 33, [x_norm, y_norm, z_norm, visibility])"
    }

    np.savez_compressed(
        cache_path,
        keypoints=keypoints,
        times=times,
        frame_idx=frame_indices,
        metadata=np.array([metadata], dtype=object)
    )

    print(f"[Cache] Saved pose cache: {cache_path}")
    print(f"[Cache] Samples: {keypoints.shape[0]}, Effective FPS: {metadata['cache_fps']:.3f}")
    return cache_path


# # ============================================================
# # 2. Detect jump candidate segments from pose cache
# # ============================================================
def detect_jump_segments_by_compact_and_rotation_from_cache(
    cache_path: str,
    compact_top_ratio: float = 0.2,   # first stage: keep top 70% most compact frames
    rot_top_ratio: float = 0.1,       # second stage: keep top 40% fastest-rotating among compact frames
    min_gap_sec: float = 0.2,
    extend_sec: float = 1.0,
    min_segment_sec: float = 0.5,
    smooth_kernel_tuck: int = 5,
    smooth_kernel_rot: int = 5,
    return_debug: bool = False,
):
    """
    Two-stage heuristic for jump candidate detection:

    Stage 1 (compactness):
        - Compute a "tucked limbs" compactness score:
            * Torso axis defined by mid-hips -> mid-shoulders.
            * Arms: wrists close to torso axis.
            * Legs: knees and ankles close to torso axis.
        - Rank frames by this compactness and keep the top compact_top_ratio.

    Stage 2 (rotation):
        - Compute body orientation angle in the image plane (using shoulders or hips).
        - Compute angular velocity |d(angle)/dt|.
        - Among the compact frames, keep the subset with highest angular velocity
          (rot_top_ratio).

    Then:
        - Group selected frames into temporal segments.
        - Extend, merge, and filter short segments.
        - Convert times back to original frame indices.
    """

    data = np.load(cache_path, allow_pickle=True)
    keypoints = data["keypoints"]        # (T, 33, 4)
    times = data["times"]                # (T,)
    metadata = data["metadata"][0]
    orig_fps = metadata["orig_fps"]
    total_frames = metadata["total_frames"]

    T = keypoints.shape[0]
    if T == 0:
        print("[Detect-combo] Empty cache, no segments.")
        return []

    # -------------------------
    # Helper: distance p->line
    # -------------------------
    def point_line_distance(p, a, b):
        """
        p, a, b: np.array([x, y])
        returns scalar distance from p to infinite line through a-b
        """
        ab = b - a
        ap = p - a
        denom = np.linalg.norm(ab)
        if denom < 1e-6:
            return 0.0
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        return abs(cross) / denom

    # ============================================================
    # 1) Compute "tucked limbs" compactness score for each frame
    # ============================================================

    tuck_score = np.full((T,), np.nan, dtype=np.float32)

    for t in range(T):
        kp = keypoints[t]

        def vis(j):
            return kp[j, 3] > 0.3

        # Need torso to be visible: shoulders + hips
        if not (vis(11) and vis(12) and vis(23) and vis(24)):
            continue

        # Torso points
        l_sh = kp[11, :2]
        r_sh = kp[12, :2]
        l_hip = kp[23, :2]
        r_hip = kp[24, :2]

        mid_shoulder = 0.5 * (l_sh + r_sh)
        mid_hip = 0.5 * (l_hip + r_hip)
        torso_axis_a = mid_hip
        torso_axis_b = mid_shoulder

        # Shoulder width as a natural scale
        shoulder_width = np.linalg.norm(r_sh - l_sh)
        if shoulder_width < 1e-6:
            shoulder_width = 1e-6

        # ----- Arms: wrists close to torso axis -----
        arm_scores = []
        for wrist_idx in (15, 16):  # left/right wrist
            if not vis(wrist_idx):
                continue
            p = kp[wrist_idx, :2]
            d_line = point_line_distance(p, torso_axis_a, torso_axis_b)
            norm_d = d_line / shoulder_width

            # Map distance to [0,1] (0.7 * shoulder width -> score 0)
            arm_thresh = 0.7
            s = 1.0 - norm_d / arm_thresh
            s = max(0.0, min(1.0, s))
            arm_scores.append(s)

        arm_in_score = np.mean(arm_scores) if arm_scores else np.nan

        # ----- Legs: knees + ankles close to torso axis -----
        leg_scores = []
        leg_triplets = [
            (23, 25, 27),  # left leg
            (24, 26, 28),  # right leg
        ]

        for hip_idx, knee_idx, ankle_idx in leg_triplets:
            if not (vis(hip_idx) and vis(knee_idx) and vis(ankle_idx)):
                continue

            knee_p = kp[knee_idx, :2]
            ankle_p = kp[ankle_idx, :2]

            d_knee = point_line_distance(knee_p, torso_axis_a, torso_axis_b)
            d_ankle = point_line_distance(ankle_p, torso_axis_a, torso_axis_b)
            norm_dk = d_knee / shoulder_width
            norm_da = d_ankle / shoulder_width

            leg_thresh = 0.8  # a bit more tolerant
            s_knee = 1.0 - norm_dk / leg_thresh
            s_ankle = 1.0 - norm_da / leg_thresh
            s_knee = max(0.0, min(1.0, s_knee))
            s_ankle = max(0.0, min(1.0, s_ankle))

            leg_scores.append(0.5 * s_knee + 0.5 * s_ankle)

        leg_score = np.mean(leg_scores) if leg_scores else np.nan

        components = []
        if not np.isnan(arm_in_score):
            components.append(arm_in_score)
        if not np.isnan(leg_score):
            components.append(leg_score)

        if not components:
            continue

        tuck_score[t] = float(np.mean(components))

    valid_mask_tuck = ~np.isnan(tuck_score)
    if not np.any(valid_mask_tuck):
        print("[Detect-combo] No valid tuck scores.")
        return []

    # Interpolate NaNs in tuck_score
    idx = np.arange(T)
    tuck_score[~valid_mask_tuck] = np.interp(
        idx[~valid_mask_tuck],
        idx[valid_mask_tuck],
        tuck_score[valid_mask_tuck]
    )

    # Smooth tuck_score over time
    if smooth_kernel_tuck > 1:
        k = smooth_kernel_tuck
        pad = k // 2
        padded = np.pad(tuck_score, (pad, pad), mode="edge")
        tuck_smooth = np.convolve(
            padded,
            np.ones(k, dtype=np.float32) / k,
            mode="valid"
        )
    else:
        tuck_smooth = tuck_score

    # ============================================================
    # 2) Compute angular velocity for each frame
    # ============================================================

    angles = np.full((T,), np.nan, dtype=np.float32)

    for t in range(T):
        kp = keypoints[t]

        def vis(j):
            return kp[j, 3] > 0.3

        used = False
        # Prefer shoulders
        if vis(11) and vis(12):
            l = kp[11, :2]
            r = kp[12, :2]
            used = True
        # Fallback to hips
        elif vis(23) and vis(24):
            l = kp[23, :2]
            r = kp[24, :2]
            used = True

        if used:
            vx = r[0] - l[0]
            vy = r[1] - l[1]
            angles[t] = np.arctan2(vy, vx)

    valid_mask_ang = ~np.isnan(angles)
    if not np.any(valid_mask_ang):
        print("[Detect-combo] No valid orientation angles.")
        return []

    # Interpolate NaNs in angles
    angles[~valid_mask_ang] = np.interp(
        idx[~valid_mask_ang],
        idx[valid_mask_ang],
        angles[valid_mask_ang]
    )

    # Time differences
    dt = np.diff(times)
    if np.any(dt <= 0):
        pos_dt = dt[dt > 0]
        default_dt = np.median(pos_dt) if len(pos_dt) > 0 else 1.0 / 15.0
        dt[dt <= 0] = default_dt

    # Angle differences (wrapped to [-pi, pi])
    dtheta = np.diff(angles)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    ang_vel = np.abs(dtheta) / dt  # rad/s
    ang_vel_full = np.zeros((T,), dtype=np.float32)
    ang_vel_full[1:] = ang_vel
    ang_vel_full[0] = ang_vel_full[1]

    # Smooth angular velocity
    if smooth_kernel_rot > 1:
        k = smooth_kernel_rot
        pad = k // 2
        padded = np.pad(ang_vel_full, (pad, pad), mode="edge")
        ang_vel_smooth = np.convolve(
            padded,
            np.ones(k, dtype=np.float32) / k,
            mode="valid"
        )
    else:
        ang_vel_smooth = ang_vel_full

    # ============================================================
    # 3) Two-stage selection: compactness then rotation
    # ============================================================

    # Stage 1: select most compact frames (large compact_top_ratio)
    sorted_by_tuck = np.argsort(-tuck_smooth)
    n_compact = max(1, int(round(T * compact_top_ratio)))
    compact_pool = sorted_by_tuck[:n_compact]

    # Stage 2: within compact_pool, select high angular velocity frames
    pool_ang = ang_vel_smooth[compact_pool]
    order_in_pool = np.argsort(-pool_ang)
    n_rot = max(1, int(round(len(compact_pool) * rot_top_ratio)))
    selected_pool_idx = order_in_pool[:n_rot]
    candidate_idx = compact_pool[selected_pool_idx]
    candidate_idx = np.array(sorted(candidate_idx))

    if len(candidate_idx) == 0:
        print("[Detect-combo] No candidate frames after two-stage filtering.")
        return []

    # ============================================================
    # 4) Group candidates into segments + extend/merge/filter
    # ============================================================

    segments_time = []
    current_start_t = times[candidate_idx[0]]
    last_t = times[candidate_idx[0]]

    cache_fps = 1.0 / np.median(np.diff(times))
    max_gap = 2.0 / cache_fps  # two cache intervals considered continuous

    for idx_c in candidate_idx[1:]:
        t = times[idx_c]
        if t - last_t > max_gap:
            segments_time.append((current_start_t, last_t))
            current_start_t = t
        last_t = t
    segments_time.append((current_start_t, last_t))

    merged = []
    total_time = total_frames / orig_fps

    for start_t, end_t in segments_time:
        # Extend to include preparation and landing
        start_t = max(0.0, start_t - extend_sec)
        end_t = min(total_time, end_t + extend_sec)

        if not merged:
            merged.append([start_t, end_t])
        else:
            ps, pe = merged[-1]
            if start_t - pe < min_gap_sec:
                merged[-1][1] = max(pe, end_t)
            else:
                merged.append([start_t, end_t])


            
    segments = []
    seg_time_ranges = []   

    for start_t, end_t in merged:
        if end_t - start_t < min_segment_sec:
            continue

        s_f = max(0, int(round(start_t * orig_fps)))
        e_f = min(total_frames - 1, int(round(end_t * orig_fps)))

        segments.append((s_f, e_f))
        seg_time_ranges.append((start_t, end_t))  

    print(f"[Detect-combo] Found {len(segments)} candidate segments.")
    
    # for i, (s, e) in enumerate(segments, 1):
    #     dur = (e - s + 1) / orig_fps
    #     print(f"  Segment {i}: frames [{s}, {e}] (duration={dur:.2f}s)")
        
    #     print(f"[Detect-combo] Found {len(segments)} candidate segments.")
    for i, (s, e) in enumerate(segments, 1):
        dur = (e - s + 1) / orig_fps
        print(f"  Segment {i}: frames [{s}, {e}] (duration={dur:.2f}s)")

    if return_debug:
        debug = {
            "times": times,                  
            "tuck_score": tuck_score,        
            "tuck_smooth": tuck_smooth,     
            "ang_vel_full": ang_vel_full,    
            "ang_vel_smooth": ang_vel_smooth,
            "candidate_idx": candidate_idx,  
            "compact_pool": compact_pool, 
            "times": times,
            "seg_time_ranges": np.array(seg_time_ranges),  
        }
        return segments, debug
    
    return segments


# ============================================================
# 3. Save segments as clips (mp4) and export skeleton for each clip
# ============================================================
def save_segments_as_clips(
    video_path: str,
    segments,
    clips_dir: str,
    min_clip_sec: float = 3.0,
    max_clip_sec: float = 5.0
):
    """
    Cut the original video into multiple short clips according to segments,
    and save each clip as a video file.

    - If a segment is shorter than min_clip_sec, we EXTEND it (preferably
      towards the future) to make it at least min_clip_sec long.
    - If a segment is longer than max_clip_sec, we keep ONLY the LAST
      max_clip_sec seconds of that segment (because jump is usually near the end).

    Args:
        video_path: path to full routine video.
        segments: list of (start_frame, end_frame) from detection.
        clips_dir: directory to save clips.
        min_clip_sec: minimum duration (in seconds) for each clip.
        max_clip_sec: maximum duration (in seconds) for each clip.

    Returns:
        clip_infos: list of dicts with:
            {
              "clip_path": str,
              "start_frame": int,
              "end_frame": int,
              "num_frames": int
            }
    """
    os.makedirs(clips_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    min_frames = int(round(min_clip_sec * fps))
    max_frames = int(round(max_clip_sec * fps))

    clip_infos = []
    print("\n[Clips] Saving candidate clips...")

    for i, (start_f, end_f) in enumerate(segments, 1):
        num_frames = end_f - start_f + 1
        if num_frames <= 0:
            continue

        if num_frames < min_frames:
            need = min_frames - num_frames

            new_end = min(total_frames - 1, end_f + need)
            new_start = start_f

            new_num = new_end - new_start + 1

            if new_num < min_frames:
                extra_back = min_frames - new_num
                new_start = max(0, new_start - extra_back)
                new_num = new_end - new_start + 1

            start_f, end_f = new_start, new_end
            num_frames = new_num

        if num_frames <= 0:
            continue

        if num_frames > max_frames:
            new_end = end_f - 60
            new_start = new_end - max_frames + 1
            new_start = max(0, new_start)
            new_end = min(total_frames - 1, new_end)

            start_f, end_f = new_start, new_end
            num_frames = end_f - start_f + 1

        if num_frames <= 0:
            continue

        clip_name = f"{video_name}_cand_jump{i:02d}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)

        print(f"  Clip {i}: frames [{start_f}, {end_f}] ({num_frames} frames) -> {clip_name}")

        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        fidx = start_f
        while fidx <= end_f:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            fidx += 1

        writer.release()

        clip_infos.append({
            "clip_path": clip_path,
            "start_frame": start_f,
            "end_frame": end_f,
            "num_frames": num_frames
        })

    cap.release()
    print(f"[Clips] Saved {len(clip_infos)} clips.")
    return clip_infos


def export_clip_skeletons_from_cache(
    cache_path: str,
    clip_infos,
    coords_dir: str
):
    """
    For each clip (defined by start/end original frame indices),
    slice the pose cache to get the corresponding 15fps keypoints, and
    save them into a separate npz file.

    Each npz will contain:
        keypoints: (T_clip, 33, 4)
        times:     (T_clip,)
        frame_idx: (T_clip,)
        metadata:  dict including clip info and reference to original video.
    """
    os.makedirs(coords_dir, exist_ok=True)

    data = np.load(cache_path, allow_pickle=True)
    keypoints = data["keypoints"]
    times = data["times"]
    frame_idx = data["frame_idx"]
    metadata_video = data["metadata"][0]

    print("\n[Coords] Exporting clip skeletons from cache...")

    for info in clip_infos:
        clip_path = info["clip_path"]
        s_f = info["start_frame"]
        e_f = info["end_frame"]

        # Mask cache frames that fall inside this segment
        mask = (frame_idx >= s_f) & (frame_idx <= e_f)
        if not np.any(mask):
            print(f"  [Warn] No cached frames found for clip: {clip_path}")
            continue

        kp_clip = keypoints[mask]    # (T_clip, 33, 4)
        t_clip = times[mask]
        f_clip = frame_idx[mask]

        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        coords_path = os.path.join(coords_dir, f"{clip_name}_coords_15fps.npz")

        meta_clip = {
            "clip_name": clip_name,
            "clip_path": clip_path,
            "source_video": metadata_video["video_path"],
            "orig_fps": metadata_video["orig_fps"],
            "total_frames_full_video": metadata_video["total_frames"],
            "cache_fps": metadata_video["cache_fps"],
            "segment_start_frame": int(s_f),
            "segment_end_frame": int(e_f),
            "num_cache_frames": int(kp_clip.shape[0]),
            "keypoint_format": metadata_video["keypoint_format"]
        }

        np.savez_compressed(
            coords_path,
            keypoints=kp_clip,
            times=t_clip,
            frame_idx=f_clip,
            metadata=np.array([meta_clip], dtype=object)
        )

        print(f"  Saved skeleton: {coords_path}  (T={kp_clip.shape[0]})")

    print("[Coords] Done exporting clip skeletons.")


# ============================================================
# 4. High-level pipeline: run everything
# ============================================================
def run_full_pipeline_15fps(
    video_path: str,
    output_root: str,
    cache_fps: float = 15.0,
    compact_top_ratio=0.2,   
    rot_top_ratio=0.1,
    # top_k_ratio: float = 0.30,
    max_clip_sec: float = 5.0,
    min_gap_sec=0.2,
    extend_sec=1.0,
    min_segment_sec=0.5,
    
   
):
    """
    High-level helper:

        1) Build pose cache (~15fps) for the full routine.
        2) Detect jump candidate segments from the cache.
        3) Save those segments as short clips (mp4).
        4) For each clip, export corresponding 15fps skeleton coordinates.

    Output structure:

        output_root/
            <video_name>/
                pose_cache_15fps.npz
                clips/
                    <video_name>_cand_jump01.mp4
                    ...
                clip_coords/
                    <video_name>_cand_jump01_coords_15fps.npz
                    ...

    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_root, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    cache_path = os.path.join(video_out_dir, "pose_cache_15fps.npz")

    # 1) Build pose cache
    build_pose_cache(
        video_path=video_path,
        cache_path=cache_path,
        cache_fps=cache_fps
    )

    # 2) Detect candidate segments from cachee
    segments = detect_jump_segments_by_compact_and_rotation_from_cache(
        cache_path=cache_path,
        compact_top_ratio=compact_top_ratio,  
        rot_top_ratio=rot_top_ratio,         
        min_gap_sec=min_gap_sec,
        extend_sec=extend_sec,
        min_segment_sec=min_segment_sec,
    )

    
    if not segments:
        print("[Pipeline] No candidate segments found. Stop.")
        return

    # 3) Save candidate clips
    clips_dir = os.path.join(video_out_dir, "clips")
    clip_infos = save_segments_as_clips(
    video_path=video_path,
    segments=segments,
    clips_dir=clips_dir,
    min_clip_sec=3.0,         
    max_clip_sec=max_clip_sec 
    )

    if not clip_infos:
        print("[Pipeline] No clips were saved (segments too short?). Stop.")
        return

    # 4) Export skeleton coordinates for each clip from cache
    coords_dir = os.path.join(video_out_dir, "clip_coords")
    export_clip_skeletons_from_cache(
        cache_path=cache_path,
        clip_infos=clip_infos,
        coords_dir=coords_dir
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE (15fps pose + candidate clips + clip coords)")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Output dir: {video_out_dir}")
    print("You can now use the files in 'clip_coords/' for LSTM training.")
    print("=" * 70)
    


if __name__ == "__main__":
    VIDEO_PATH = r"D:\\2025SA_WOMEN\\2025SA_WOMEN-seg11.mp4"
    OUTPUT_ROOT = r"D:\\jump_outputs"

    run_full_pipeline_15fps(
        video_path=VIDEO_PATH,
        output_root=OUTPUT_ROOT,
        cache_fps=15.0,
        compact_top_ratio=0.2,   # first stage: keep top 20% most compact frames
        rot_top_ratio=0.1,       # second stage: keep top 10% fastest-rotating among compact frames
        max_clip_sec=5.0,    # each clip <= 5s
        min_gap_sec=0.2,
        extend_sec=1.0,
        min_segment_sec=0.5,
        

    )
