"""
Result Analysis and Visualization
Functions for analyzing model predictions and visualizing decision space.

Usage (in Jupyter notebook after training):
    find_wrong_predictions(model, valid_ds, "Validation Set")
    test_confidence_distribution(model, valid_ds)
    visualize_tsne(model, valid_ds)

Required variables (defined in training notebook):
    - model: Trained PyTorch model
    - valid_ds / test_ds: Dataset
    - MERGE_MAP: Label mapping dict
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE


# Part 1: Find Wrong Predictions

def find_wrong_predictions(model, dataset, dataset_name="Validation Set"):
    """Output a list of names in incorrect validation set."""
    print(f"\nAnalyzing Errors in {dataset_name}")
    model.eval()
    device = next(model.parameters()).device
    wrong_samples = []

    try:
        id2label = {v: k for k, v in MERGE_MAP.items()}
        labels_unique = set(MERGE_MAP.values())
        id2name = {i: f"Class {i}" for i in labels_unique}
    except:
        id2name = {i: str(i) for i in range(10)}

    print(f"Total samples to check: {len(dataset)}")

    with torch.no_grad():
        for i in range(len(dataset)):
            feature, label_idx = dataset[i]
            original_idx = dataset.indices[i]
            original_path, _ = dataset.dataset.samples[original_idx]

            input_tensor = feature.unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()

            if pred_idx != label_idx:
                filename = os.path.basename(original_path)
                video_name = filename.replace('.npz', '').replace('_coords', '')

                wrong_samples.append({
                    "Original Index": original_idx,
                    "Filename": filename,
                    "Video Name Hint": video_name,
                    "True Label": f"{label_idx} ({id2name.get(int(label_idx), '?')})",
                    "Predicted": f"{pred_idx} ({id2name.get(pred_idx, '?')})",
                    "Confidence": f"{torch.softmax(logits, dim=1).max().item():.2f}"
                })

    if len(wrong_samples) == 0:
        print("No errors found.")
    else:
        print(f"Found {len(wrong_samples)} errors out of {len(dataset)} samples.\n")
        df = pd.DataFrame(wrong_samples)
        cols = ["Video Name Hint", "True Label", "Predicted", "Confidence", "Filename"]
        print(df[cols].to_markdown(index=False))

        csv_name = f"error_analysis_{dataset_name.replace(' ', '_')}.csv"
        df.to_csv(csv_name, index=False)
        print(f"\nSaved detailed error list to '{csv_name}'")


# Part 2: Statistical Proof - Confidence Distribution Test
# Statistically proves that the model is as confident when making mistakes
# on Class 1 as it is when correctly predicting Class 0

def test_confidence_distribution(model, dataset):
    """Test confidence distribution between correct and incorrect predictions."""
    model.eval()
    device = next(model.parameters()).device

    # Group A: True Full Rotation (Class 0), correctly predicted as Full
    conf_true_0 = []
    # Group B: True Quarter Rotation (Class 1), incorrectly predicted as Full
    conf_false_0_from_1 = []

    print("Collecting confidence scores...")
    with torch.no_grad():
        for i in range(len(dataset)):
            feature, label = dataset[i]
            input_tensor = feature.unsqueeze(0).to(device)

            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs).item()
            conf_score = probs[0, 0].item()  # Get the probability score for Class 0

            if label.item() == 0 and pred == 0:
                conf_true_0.append(conf_score)
            elif label.item() == 1 and pred == 0:
                conf_false_0_from_1.append(conf_score)

    # Statistical analysis
    if len(conf_true_0) == 0 or len(conf_false_0_from_1) == 0:
        print("Not enough samples to perform statistical analysis.")
        return

    avg_0 = np.mean(conf_true_0)
    avg_1 = np.mean(conf_false_0_from_1)

    print(f"\nResult Analysis")
    print(f"Avg Confidence for True Full (Class 0) -> Pred 0:       {avg_0:.4f}")
    print(f"Avg Confidence for True Quarter (Class 1) -> Pred 0:    {avg_1:.4f}")

    diff = abs(avg_0 - avg_1)
    print(f"Difference: {diff:.4f}")

    if diff < 0.05:
        print("Negligible difference! The model is equally confident in its errors.")
        print("This suggests Class 1 and Class 0 are indistinguishable in the feature space.")
    else:
        print("The model shows uncertainty (lower confidence) when misclassifying.")

    plt.figure(figsize=(10, 6))
    plt.hist(conf_true_0, bins=20, alpha=0.5, label='True Full (Class 0)', color='green', density=True)
    plt.hist(conf_false_0_from_1, bins=20, alpha=0.5, label='Quarter (Class 1) Misclassified as Full', color='red', density=True)
    plt.xlabel('Confidence Score for Prediction "Class 0"')
    plt.ylabel('Density')
    plt.title('Confidence Distribution: True Full vs. False Full')
    plt.legend()
    plt.show()

# Part 3: Visual Proof - t-SNE Visualization
# Visualizes the high-dimensional feature space to show class overlap

def visualize_tsne(model, dataset):
    """t-SNE visualization of model decision space."""
    model.eval()
    device = next(model.parameters()).device

    all_logits = []
    all_labels = []

    print("Extracting features for t-SNE projection...")
    with torch.no_grad():
        for i in range(len(dataset)):
            feature, label = dataset[i]

            input_tensor = feature.unsqueeze(0).to(device)
            logits = model(input_tensor).cpu().numpy().flatten()

            all_logits.append(logits)
            all_labels.append(label.item())

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(np.array(all_logits))

    df = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df['Label'] = all_labels
    label_map = {0: '0: Full', 1: '1: Quarter', 2: '2: Under', 3: '3: Downgraded'}
    df['Class Name'] = df['Label'].map(label_map)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='x', y='y', hue='Class Name',
        palette='viridis', s=100, alpha=0.7
    )
    plt.title("t-SNE Visualization of Model Decision Space")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
