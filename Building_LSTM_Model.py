'''
Build a two-branch LSTM network for jumps & non-jumps classification.
The network combines both skeleton data and specific features, achieving 90% accuracy

The feature list:
      - hip_up: hip height relative to baseline
      - ankle_up: ankle height relative to baseline
      - compact: how close arms/legs are to the body center
      - straight: trunk verticality
      - arms_up: hand height relative to shoulders
      - arms_front: hand forward distance to shoulders
      - abs_omega: absolute angular velocity (yaw rate)
      - ankle_angle: knee-ankle-toe angle in degrees (bigger in jumps, ~<=90 on ice)
'''


import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. Feature utilities
# =========================

def normalize_pose_xy(kp):
    """
    Normalize 2D pose coordinates:
      - Use mid-hip as origin.
      - Scale by hip-shoulder distance as body size.
    Args:
        kp: (T, 33, 4) MediaPipe keypoints (x, y, z, visibility), normalized [0,1].
    Returns:
        pose_norm: (T, 33*2) flattened (x, y) for all joints.
    """
    T, J, _ = kp.shape
    xy = kp[..., :2].copy()          # (T, 33, 2)
    vis = kp[..., 3]                 # (T, 33)

    # mid-hip
    lh = xy[:, 23, :]  # left hip
    rh = xy[:, 24, :]  # right hip
    mid_hip = 0.5 * (lh + rh)  # (T, 2)

    # mid-shoulder
    ls = xy[:, 11, :]  # left shoulder
    rs = xy[:, 12, :]  # right shoulder
    mid_sh = 0.5 * (ls + rs)  # (T, 2)

    # body vector and length (used as scale)
    body_vec = mid_sh - mid_hip  # (T, 2)
    body_len = np.linalg.norm(body_vec, axis=-1)  # (T,)

    # avoid zero scale
    if np.any(body_len > 1e-4):
        avg_body_len = np.mean(body_len[body_len > 1e-4])
    else:
        avg_body_len = 1.0
    body_len[body_len < 1e-4] = avg_body_len

    # center at mid-hip
    xy_centered = xy - mid_hip[:, None, :]          # (T, 33, 2)
    xy_norm = xy_centered / body_len[:, None, None] # (T, 33, 2)

    pose_norm = xy_norm.reshape(T, -1).astype(np.float32)  # (T, 33*2)
    return pose_norm


def compute_body_yaw(kp):
    """
    Estimate body yaw in image plane using mid-hip -> mid-shoulder vector.

    Args:
        kp: (T, 33, 4) keypoints.
    Returns:
        yaw: (T,) angles in radians, in [-pi, pi].
    """
    xy = kp[..., :2]  # (T, 33, 2)
    lh = xy[:, 23, :]
    rh = xy[:, 24, :]
    ls = xy[:, 11, :]
    rs = xy[:, 12, :]

    mid_hip = 0.5 * (lh + rh)
    mid_sh = 0.5 * (ls + rs)

    vec = mid_sh - mid_hip  # (T, 2)
    vx, vy = vec[:, 0], vec[:, 1]

    # Define "up" (negative y) as 0 angle; horizontal rotations change yaw.
    yaw = np.arctan2(vx, -vy)  # [-pi, pi]
    return yaw.astype(np.float32)


def compute_clip_features(kp, times):
    """
    Compute per-frame engineered features for one clip:
      - hip_up: hip height relative to baseline
      - ankle_up: ankle height relative to baseline
      - compact: how close arms/legs are to body center
      - straight: trunk verticality
      - arms_up: hand height relative to shoulders
      - arms_front: hand forward distance to shoulders
      - abs_omega: absolute angular velocity (yaw rate)
      - ankle_angle: knee-ankle-toe angle in degrees (bigger in jumps, ~<=90 on ice)

    Args:
        kp: (T, 33, 4) keypoints.
        times: (T,) timestamps in seconds.
    Returns:
        feat: (T, D_feat) float32
        feat_names: list of feature names
    """
    T = kp.shape[0]
    xy = kp[..., :2]        # (T, 33, 2)
    vis = kp[..., 3]        # (T, 33)

    # --- hip & ankle heights (image y) ---
    hip_y = 0.5 * (xy[:, 23, 1] + xy[:, 24, 1])     # (T,)
    ankle_y = 0.5 * (xy[:, 27, 1] + xy[:, 28, 1])   # (T,)

    hip_base = np.median(hip_y)
    ankle_base = np.median(ankle_y)

    hip_up = hip_base - hip_y       # >0: higher than baseline
    ankle_up = ankle_base - ankle_y

    # --- compactness: distance of wrists/ankles/knees to mid-hip ---
    mid_hip = 0.5 * (xy[:, 23, :] + xy[:, 24, :])   # (T, 2)
    mid_sh = 0.5 * (xy[:, 11, :] + xy[:, 12, :])   # (T, 2)

    # body length for normalization
    body_vec = mid_sh - mid_hip          # (T, 2)
    body_len = np.linalg.norm(body_vec, axis=-1)  # (T,)
    if np.any(body_len > 1e-4):
        avg_body_len = np.mean(body_len[body_len > 1e-4])
    else:
        avg_body_len = 1.0
    body_len[body_len < 1e-4] = avg_body_len

    key_ids = [15, 16, 27, 28, 25, 26]   # wrists + ankles + knees
    pts = xy[:, key_ids, :]              # (T, 6, 2)
    dist = np.linalg.norm(pts - mid_hip[:, None, :], axis=-1)  # (T, 6)
    dist_norm = dist / body_len[:, None]                         # (T, 6)
    compact = 1.0 - np.clip(np.mean(dist_norm, axis=-1), 0.0, 2.0)  # higher = more compact

    # --- straightness: alignment of hip->shoulder with vertical direction ---
    v = body_vec   # (T, 2)
    v_norm = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
    up_vec = np.array([0.0, -1.0], dtype=np.float32)
    straight = (v_norm @ up_vec).astype(np.float32)  # 1 ~ vertical, 0 ~ horizontal

    # --- arms up / arms front ---
    lw, rw = xy[:, 15, :], xy[:, 16, :]  # wrists
    ls, rs = xy[:, 11, :], xy[:, 12, :]  # shoulders

    # shoulder_y - wrist_y: larger = wrist higher above shoulder
    arms_up_score = 0.5 * ((ls[:, 1] - lw[:, 1]) + (rs[:, 1] - rw[:, 1]))
    # horizontal distance to shoulders (proxy for arms forward)
    arms_front_score = 0.5 * (np.abs(lw[:, 0] - ls[:, 0]) + np.abs(rw[:, 0] - rs[:, 0]))

    # --- angular velocity based on body yaw ---
    yaw = compute_body_yaw(kp)          # (T,)
    yaw_unwrap = np.unwrap(yaw)         # remove 2π jumps

    dt = np.diff(times)
    # avoid zero dt
    valid_dt = dt[dt > 1e-3]
    if valid_dt.size > 0:
        default_dt = np.median(valid_dt)
    else:
        default_dt = 1.0 / 15.0
    dt[dt <= 1e-3] = default_dt

    omega = np.zeros_like(yaw_unwrap)
    omega[1:] = np.diff(yaw_unwrap) / dt
    abs_omega = np.abs(omega)

    # --- ankle angle: knee-ankle-toe angle (degrees) ---
    # MediaPipe indices:
    #   left knee=25, right knee=26, left ankle=27, right ankle=28,
    #   left foot index=31, right foot index=32
    lk, rk = xy[:, 25, :], xy[:, 26, :]
    la, ra = xy[:, 27, :], xy[:, 28, :]
    lt, rt = xy[:, 31, :], xy[:, 32, :]

    def joint_angle(p_prox, p_joint, p_dist):
        """
        Angle at p_joint formed by p_prox - p_joint and p_dist - p_joint, in degrees.
        p_prox, p_joint, p_dist: (T, 2)
        """
        v1 = p_prox - p_joint
        v2 = p_dist - p_joint
        v1_norm = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6)
        cosang = np.clip(np.sum(v1_norm * v2_norm, axis=-1), -1.0, 1.0)
        ang = np.arccos(cosang)  # radians
        return ang * (180.0 / np.pi)

    left_ankle_angle  = joint_angle(lk, la, lt)   # (T,)
    right_ankle_angle = joint_angle(rk, ra, rt)   # (T,)

    ankle_angle = np.maximum(left_ankle_angle, right_ankle_angle).astype(np.float32)

    feat_list = [
        hip_up,
        ankle_up,
        compact,
        straight,
        arms_up_score,
        arms_front_score,
        abs_omega,
        ankle_angle,
    ]
    feat_names = [
        "hip_up",
        "ankle_up",
        "compact",
        "straight",
        "arms_up",
        "arms_front",
        "abs_omega",
        "ankle_angle",
    ]

    feat = np.stack(feat_list, axis=-1).astype(np.float32)  # (T, D_feat)
    return feat, feat_names


# =========================
# 2. Dataset and collate_fn
# =========================

class FSClipsDataset(Dataset):
    """
    Dataset to read npz clips from:

        root_dir/
            jumps/*.npz
            non-jumps/*.npz

    Each npz is expected to contain:
        keypoints: (T, 33, 4)
        times:     (T,)

    This dataset performs:
        - Temporal feature extraction.
        - Pose normalization.
        - Train/val split by random shuffling.
    """
    def __init__(self, root_dir, split="train", val_fraction=0.2, random_seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.samples = []

        jumps_dir = os.path.join(root_dir, "jumps")
        non_dir = os.path.join(root_dir, "non-jumps")

        jump_files = []
        if os.path.isdir(jumps_dir):
            jump_files = [
                os.path.join(jumps_dir, f)
                for f in os.listdir(jumps_dir)
                if f.lower().endswith(".npz")
            ]
        non_files = []
        if os.path.isdir(non_dir):
            non_files = [
                os.path.join(non_dir, f)
                for f in os.listdir(non_dir)
                if f.lower().endswith(".npz")
            ]

        jump_files.sort()
        non_files.sort()

        print(f"[Dataset] Found {len(jump_files)} jump clips, {len(non_files)} non-jump clips")

        # Make positive and negative counts balanced
        n = min(len(jump_files), len(non_files))
        jump_files = jump_files[:n]
        non_files = non_files[:n]

        labels = [1] * len(jump_files) + [0] * len(non_files)
        paths = jump_files + non_files

        # Shuffle and split into train/val
        random.seed(random_seed)
        idx_all = list(range(len(paths)))
        random.shuffle(idx_all)

        n_total = len(paths)
        n_val = int(round(n_total * val_fraction))
        val_idx = set(idx_all[:n_val])
        train_idx = set(idx_all[n_val:])

        for i in range(n_total):
            item = {"path": paths[i], "label": labels[i]}
            if i in train_idx and split == "train":
                self.samples.append(item)
            elif i in val_idx and split == "val":
                self.samples.append(item)

        print(f"[Dataset] {split} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        npz_path = meta["path"]
        label = meta["label"]

        data = np.load(npz_path, allow_pickle=True)
        kp = data["keypoints"].astype(np.float32)   # (T, 33, 4)
        times = data["times"].astype(np.float32)    # (T,)

        T = kp.shape[0]
        pose_seq = normalize_pose_xy(kp)                 # (T, 33*2)
        feat_seq, _ = compute_clip_features(kp, times)   # (T, D_feat)

        return {
            "pose_seq": pose_seq,
            "feat_seq": feat_seq,
            "length": T,
            "label": label,
            "path": npz_path
        }


def collate_fn(batch, max_len=80):
    """
    Collate function for variable-length sequences:
      - Truncate sequences longer than max_len.
      - Zero-pad sequences shorter than max_len.
    Args:
        batch: list of items from FSClipsDataset.
        max_len: maximum sequence length for padding.
    Returns:
        pose_batch: (B, max_len, Dp)
        feat_batch: (B, max_len, Df)
        lengths:    (B,) actual lengths before padding/truncation
        labels:     (B,)
    """
    batch_size = len(batch)
    Dp = batch[0]["pose_seq"].shape[1]
    Df = batch[0]["feat_seq"].shape[1]

    pose_batch = np.zeros((batch_size, max_len, Dp), dtype=np.float32)
    feat_batch = np.zeros((batch_size, max_len, Df), dtype=np.float32)
    lengths = []
    labels = []

    for i, item in enumerate(batch):
        pose_seq = item["pose_seq"]
        feat_seq = item["feat_seq"]
        T = item["length"]
        L = min(T, max_len)

        pose_batch[i, :L, :] = pose_seq[:L]
        feat_batch[i, :L, :] = feat_seq[:L]
        lengths.append(L)
        labels.append(item["label"])

    pose_batch = torch.from_numpy(pose_batch)   # (B, L, Dp)
    feat_batch = torch.from_numpy(feat_batch)   # (B, L, Df)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    return pose_batch, feat_batch, lengths, labels

# =========================
# 3. Two-branch LSTM model
# =========================

class TwoBranchLSTM(nn.Module):
    """
    Two-branch LSTM:
      - Branch 1: processes normalized joint coordinates.
      - Branch 2: processes engineered features.
      - Final head: merges both branches and predicts jump / non-jump.
    """
    def __init__(
        self,
        pose_dim,
        feat_dim,
        pose_hidden=128,
        feat_hidden=64,
        num_layers=1,
        dropout=0.1
    ):
        super().__init__()
        self.pose_lstm = nn.LSTM(
            input_size=pose_dim,
            hidden_size=pose_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.feat_lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        combined_dim = pose_hidden * 2 + feat_hidden * 2

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # output logit
        )

    def forward(self, pose_seq, feat_seq, lengths):
        """
        Args:
            pose_seq: (B, L, Dp)
            feat_seq: (B, L, Df)
            lengths:  (B,)
        Returns:
            logits: (B,) unnormalized scores for BCEWithLogitsLoss
        """
        # Sort by length for pack_padded_sequence
        lengths_sorted, idx_sorted = torch.sort(lengths, descending=True)
        pose_sorted = pose_seq[idx_sorted]
        feat_sorted = feat_seq[idx_sorted]

        packed_pose = nn.utils.rnn.pack_padded_sequence(
            pose_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_feat = nn.utils.rnn.pack_padded_sequence(
            feat_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )

        _, (h_pose, _) = self.pose_lstm(packed_pose)  # (num_layers*2, B, H_pose)
        _, (h_feat, _) = self.feat_lstm(packed_feat)  # (num_layers*2, B, H_feat)

        # Take last layer's forward and backward hidden states
        h_pose_last = torch.cat([h_pose[-2], h_pose[-1]], dim=-1)  # (B, 2*H_pose)
        h_feat_last = torch.cat([h_feat[-2], h_feat[-1]], dim=-1)  # (B, 2*H_feat)

        # Restore original order
        _, idx_unsort = torch.sort(idx_sorted)
        h_pose_last = h_pose_last[idx_unsort]
        h_feat_last = h_feat_last[idx_unsort]

        h = torch.cat([h_pose_last, h_feat_last], dim=-1)  # (B, 2H_pose + 2H_feat)
        logit = self.head(h).squeeze(-1)                   # (B,)
        return logit

# =========================
# 4. Train / eval loops
# =========================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for pose_seq, feat_seq, lengths, labels in loader:
        pose_seq = pose_seq.to(device)
        feat_seq = feat_seq.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pose_seq, feat_seq, lengths)   # (B,)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for pose_seq, feat_seq, lengths, labels in loader:
            pose_seq = pose_seq.to(device)
            feat_seq = feat_seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(pose_seq, feat_seq, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


# =========================
# 5. Test / new data evaluation
# =========================
def collate_fn_with_path(batch, max_len=64):
    """
    Collate function similar to collate_fn, but also returns the file paths.

    Returns:
        pose_batch: (B, max_len, Dp)
        feat_batch: (B, max_len, Df)
        lengths:    (B,)
        labels:     (B,)
        paths:      list of str, length B
    """
    batch_size = len(batch)
    Dp = batch[0]["pose_seq"].shape[1]
    Df = batch[0]["feat_seq"].shape[1]

    pose_batch = np.zeros((batch_size, max_len, Dp), dtype=np.float32)
    feat_batch = np.zeros((batch_size, max_len, Df), dtype=np.float32)
    lengths = []
    labels = []
    paths = []

    for i, item in enumerate(batch):
        pose_seq = item["pose_seq"]
        feat_seq = item["feat_seq"]
        T = item["length"]
        L = min(T, max_len)

        pose_batch[i, :L, :] = pose_seq[:L]
        feat_batch[i, :L, :] = feat_seq[:L]
        lengths.append(L)
        labels.append(item["label"])
        paths.append(item["path"])

    pose_batch = torch.from_numpy(pose_batch)
    feat_batch = torch.from_numpy(feat_batch)
    lengths    = torch.tensor(lengths, dtype=torch.long)
    labels     = torch.tensor(labels, dtype=torch.float32)

    return pose_batch, feat_batch, lengths, labels, paths


def evaluate_on_labeled_test_set(
    model,
    test_root,
    device,
    max_len=64,
    batch_size=16,
    threshold=0.5,
    max_show_tp_tn=5,
):
    """
    Evaluate model on a labeled test set and print:
      - accuracy
      - confusion matrix
      - example clips for all four cases: TP, TN, FP, FN

    test_root structure:
        test_root/
            jumps/*.npz       (label = 1)
            non-jumps/*.npz   (label = 0)
    """
    model.eval()

    # Use FSClipsDataset; with val_fraction=0.0 we get all samples in 'train' split.
    test_dataset = FSClipsDataset(
        root_dir=test_root,
        split="train",
        val_fraction=0.0,
        random_seed=123,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_with_path(b, max_len=max_len),
    )

    all_labels = []
    all_probs = []
    all_paths = []

    # Four buckets
    tps = []  # true=1, pred=1
    tns = []  # true=0, pred=0
    fps = []  # true=0, pred=1
    fns = []  # true=1, pred=0

    with torch.no_grad():
        for pose_batch, feat_batch, lengths, labels, paths in test_loader:
            pose_batch = pose_batch.to(device)
            feat_batch = feat_batch.to(device)
            lengths    = lengths.to(device)
            labels     = labels.to(device)

            logits = model(pose_batch, feat_batch, lengths)  # (B,)
            probs  = torch.sigmoid(logits)                   # (B,)

            probs_np  = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np  = (probs_np >= threshold).astype(np.float32)

            for y_true, y_pred, p, path in zip(labels_np, preds_np, probs_np, paths):
                all_labels.append(y_true)
                all_probs.append(p)
                all_paths.append(path)

                if y_true == 1.0 and y_pred == 1.0:
                    tps.append((path, y_true, p))
                elif y_true == 0.0 and y_pred == 0.0:
                    tns.append((path, y_true, p))
                elif y_true == 0.0 and y_pred == 1.0:
                    fps.append((path, y_true, p))
                elif y_true == 1.0 and y_pred == 0.0:
                    fns.append((path, y_true, p))

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds_all  = (all_probs >= threshold).astype(np.float32)

    # ----- accuracy -----
    accuracy = (preds_all == all_labels).mean() if len(all_labels) > 0 else 0.0
    print(f"\n[Test] Labeled test set size: {len(all_labels)}")
    print(f"[Test] Accuracy: {accuracy:.3f}")

    # ----- confusion matrix -----
    tn = np.sum((all_labels == 0) & (preds_all == 0))
    fp = np.sum((all_labels == 0) & (preds_all == 1))
    fn = np.sum((all_labels == 1) & (preds_all == 0))
    tp = np.sum((all_labels == 1) & (preds_all == 1))

    print("\n[Confusion Matrix] (rows = true, cols = predicted)")
    print("                pred_non-jump   pred_jump")
    print(f"true_non-jump      {tn:5d}          {fp:5d}")
    print(f"true_jump          {fn:5d}          {tp:5d}")

    # extra stats
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    print(f"\n[Stats] Precision (jump class): {precision:.3f}")
    print(f"[Stats] Recall    (jump class): {recall:.3f}")

    # ----- examples: TP / TN / FP / FN -----

    # True Positives (correct jump)
    print(f"\n[Examples] True Positives (true=jump, pred=jump) (up to {max_show_tp_tn}):")
    for i, (path, y_true, p) in enumerate(tps[:max_show_tp_tn], start=1):
        print(f"  TP {i:02d}. {path}")
        print(f"       prob_jump={p:.3f}")

    # True Negatives (correct non-jump)
    print(f"\n[Examples] True Negatives (true=non-jump, pred=non-jump) (up to {max_show_tp_tn}):")
    for i, (path, y_true, p) in enumerate(tns[:max_show_tp_tn], start=1):
        print(f"  TN {i:02d}. {path}")
        print(f"       prob_jump={p:.3f}")

    # False Positives: show ALL (these are most interesting)
    print(f"\n[Examples] False Positives (true=non-jump, pred=jump) (show ALL, count={len(fps)}):")
    for i, (path, y_true, p) in enumerate(fps, start=1):
        print(f"  FP {i:02d}. {path}")
        print(f"       prob_jump={p:.3f}")

    # False Negatives: show ALL (also very important)
    print(f"\n[Examples] False Negatives (true=jump, pred=non-jump) (show ALL, count={len(fns)}):")
    for i, (path, y_true, p) in enumerate(fns, start=1):
        print(f"  FN {i:02d}. {path}")
        print(f"       prob_jump={p:.3f}")

    return accuracy, (tn, fp, fn, tp), {"TP": tps, "TN": tns, "FP": fps, "FN": fns}


def predict_on_unlabeled_folder(
    model,
    folder,
    device,
    max_len=64,
    threshold=0.5,
):
    """
    Run the trained model on a folder of unlabeled npz clips and
    print predicted labels (jump / non-jump) with probabilities.

    Folder structure:
        folder/
            *.npz   (each contains keypoints, times)
    """
    model.eval()
    folder = os.path.abspath(folder)
    print(f"[Predict] Scanning folder: {folder}")

    npz_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".npz")
    ]
    npz_files.sort()

    if not npz_files:
        print("[Predict] No npz files found.")
        return

    for fname in npz_files:
        path = os.path.join(folder, fname)
        data = np.load(path, allow_pickle=True)
        kp    = data["keypoints"].astype(np.float32)  # (T, 33, 4)
        times = data["times"].astype(np.float32)      # (T,)

        # Compute pose & features the same way as in dataset
        pose_seq = normalize_pose_xy(kp)               # (T, Dp)
        feat_seq, _ = compute_clip_features(kp, times) # (T, Df)
        T = pose_seq.shape[0]
        L = min(T, max_len)

        # Prepare padded tensors of shape (1, max_len, D)
        Dp = pose_seq.shape[1]
        Df = feat_seq.shape[1]

        pose_batch = np.zeros((1, max_len, Dp), dtype=np.float32)
        feat_batch = np.zeros((1, max_len, Df), dtype=np.float32)
        pose_batch[0, :L, :] = pose_seq[:L]
        feat_batch[0, :L, :] = feat_seq[:L]

        lengths = torch.tensor([L], dtype=torch.long)

        pose_batch = torch.from_numpy(pose_batch).to(device)
        feat_batch = torch.from_numpy(feat_batch).to(device)
        lengths    = lengths.to(device)

        with torch.no_grad():
            logit = model(pose_batch, feat_batch, lengths)  # (1,)
            prob  = torch.sigmoid(logit).item()

        label = "jump" if prob >= threshold else "non-jump"
        print(f"[Predict] {fname}: label={label}, prob_jump={prob:.3f}")
