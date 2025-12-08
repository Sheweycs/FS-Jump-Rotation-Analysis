"""
Training and Evaluation Functions

This module provides functions for training and evaluating TCN models.

Usage Example:
    from training import train_one_epoch, evaluate
    
    # Training
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    
    # Evaluation (validation only)
    va_loss, va_acc, f1_m, preds, gts, recall_m, _, _ = evaluate(model, valid_loader, device)
    
    # Evaluation with training metrics
    va_loss, va_acc, f1_m, preds, gts, recall_m, tr_loss, tr_acc = evaluate(
        model, valid_loader, device, train_loader=train_loader
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, optimizer, criterion, device=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on (e.g., 'cuda' or 'cpu').
                If None, will be inferred from model parameters.
    
    Returns:
        Average training loss for the epoch
    """
    if device is None:
        # Infer device from model parameters
        device = next(model.parameters()).device
    
    model.train()
    total_loss, n = 0.0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device=None, train_loader=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data (validation/test)
        device: Device to run evaluation on (e.g., 'cuda' or 'cpu'). 
                If None, will be inferred from model parameters.
        train_loader: Optional DataLoader for training data. If provided,
                      training loss and accuracy will also be computed.
    
    Returns:
        tuple: (avg_loss, acc, f1_m, preds, gts, recall_m, train_loss, train_acc)
            - avg_loss: Average loss on evaluation data
            - acc: Accuracy on evaluation data
            - f1_m: Macro-averaged F1 score
            - preds: List of predictions
            - gts: List of ground truth labels
            - recall_m: Macro-averaged recall
            - train_loss: Training loss (None if train_loader not provided)
            - train_acc: Training accuracy (None if train_loader not provided)
    """
    if device is None:
        # Infer device from model parameters
        device = next(model.parameters()).device
    
    model.eval()
    preds, gts, loss_sum, n = [], [], 0.0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        n += x.size(0)
        pred = logits.softmax(-1).argmax(-1).cpu().numpy().tolist()
        preds += pred
        gts += y.cpu().numpy().tolist()
    
    avg_loss = loss_sum / max(1, n)
    acc = (np.array(preds) == np.array(gts)).mean()
    f1m = f1_score(gts, preds, average="macro")
    recall_m = recall_score(gts, preds, average="macro")   # macro recall

    # Compute training metrics if train_loader is provided
    train_loss, train_acc = None, None
    if train_loader is not None:
        tr_preds, tr_gts, tr_loss_sum, tr_n = [], [], 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            tr_loss_sum += loss.item()
            tr_n += x.size(0)
            pred = logits.softmax(-1).argmax(-1).cpu().numpy().tolist()
            tr_preds += pred
            tr_gts += y.cpu().numpy().tolist()
        
        train_loss = tr_loss_sum / max(1, tr_n)
        train_acc = (np.array(tr_preds) == np.array(tr_gts)).mean()

    return avg_loss, acc, f1m, preds, gts, recall_m, train_loss, train_acc


def plot_training_curves(history):
    """
    history: dict with keys:
        train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- Loss Curve ----
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Accuracy Curve ----
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
