"""
Training and Evaluation Functions

This module provides functions for training and evaluating TCN models.

Usage Example:
    from training import train_one_epoch, evaluate
    
    # Training
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    
    # Evaluation
    avg_loss, acc, f1_m, preds, gts, recall_m = evaluate(model, valid_loader, device)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, recall_score


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
def evaluate(model, loader, device=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data
        device: Device to run evaluation on (e.g., 'cuda' or 'cpu'). 
                If None, will be inferred from model parameters.
    
    Returns:
        tuple: (avg_loss, acc, f1_m, preds, gts, recall_m)
            - avg_loss: Average loss
            - acc: Accuracy
            - f1_m: Macro-averaged F1 score
            - preds: List of predictions
            - gts: List of ground truth labels
            - recall_m: Macro-averaged recall
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

    return avg_loss, acc, f1m, preds, gts, recall_m

