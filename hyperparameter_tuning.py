"""
Hyperparameter Tuning Module for TCN Model

This module provides functions for hyperparameter tuning using Optuna's Bayesian Optimization.

Usage Example:
    from tcn_model import TinyTCN
    from training import train_one_epoch, evaluate
    from hyperparameter_tuning import create_objective_f1, run_hyperparameter_tuning
    
    # Create objective function
    objective = create_objective_f1(
        model_class=TinyTCN,
        c_in=132,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        train_loader=train_loader,
        valid_loader=valid_loader,
        class_weights=class_weights,
        train_one_epoch=train_one_epoch,
        evaluate=evaluate,
        max_epochs=15
    )
    
    # Run hyperparameter tuning
    study = run_hyperparameter_tuning(objective, n_trials=100)
    
    # Access best parameters
    best_params = study.best_params
    best_value = study.best_value
"""

import optuna
import torch
import torch.nn as nn


def train_for_optuna(model, train_loader, valid_loader, lr, weight_decay, p_drop, 
                     class_weights, train_one_epoch, evaluate, device, max_epochs=15):
    """
    Train model for a single Optuna trial using F1 score as the metric.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        p_drop: Dropout probability
        class_weights: Class weights for loss function
        train_one_epoch: Function to train one epoch
        evaluate: Function to evaluate the model
        device: Device to run training on
        max_epochs: Maximum number of epochs to train
    
    Returns:
        best_f1: Best F1 score achieved during training
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1
    for epoch in range(1, max_epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _, va_recall = evaluate(model, valid_loader, device)

        if va_f1 > best_f1:
            best_f1 = va_f1

    return best_f1


def train_for_optuna_r(model, train_loader, valid_loader, lr, weight_decay, p_drop,
                       class_weights, train_one_epoch, evaluate, device, max_epochs=15):
    """
    Train model for a single Optuna trial using recall as the metric.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        p_drop: Dropout probability
        class_weights: Class weights for loss function
        train_one_epoch: Function to train one epoch
        evaluate: Function to evaluate the model
        device: Device to run training on
        max_epochs: Maximum number of epochs to train
    
    Returns:
        best_recall: Best recall score achieved during training
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_recall = -1
    for epoch in range(1, max_epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _, va_recall = evaluate(model, valid_loader, device)

        if va_recall > best_recall:
            best_recall = va_recall

    return best_recall


def create_objective_f1(model_class, c_in, num_classes, device, train_loader, valid_loader,
                        class_weights, train_one_epoch, evaluate, max_epochs=15,
                        hidden_dim_options=[64, 128, 256], kernel_size_options=[3, 5, 7],
                        tune_kernel_size=False):
    """
    Create an objective function for Optuna optimization using F1 score.
    
    Args:
        model_class: The model class to instantiate (e.g., TinyTCN)
        c_in: Input channel dimension
        num_classes: Number of classes
        device: Device to run training on
        train_loader: Training data loader
        valid_loader: Validation data loader
        class_weights: Class weights for loss function
        train_one_epoch: Function to train one epoch
        evaluate: Function to evaluate the model
        max_epochs: Maximum number of epochs per trial
        hidden_dim_options: List of hidden dimension options to search
        kernel_size_options: List of kernel size options to search
        tune_kernel_size: Whether to tune kernel size (if False, kernel_size is ignored)
    
    Returns:
        objective: Function that can be passed to study.optimize()
    """
    def objective(trial):
        # Search space
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        p_drop = trial.suggest_float("p_drop", 0.0, 0.5)
        
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_options)
        kernel_size = trial.suggest_categorical("kernel_size", kernel_size_options)

        print(f"\n==============================")
        print(f"🔥 Starting Trial {trial.number}")
        print(f"Hyperparameters:")
        print(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}, p_drop={p_drop:.3f}")
        print(f"  hidden_dim={hidden_dim}, kernel_size={kernel_size}")
        print(f"==============================")

        model = model_class(c_in=c_in, num_classes=num_classes, p_drop=p_drop).to(device)

        # Optionally modify kernel size if tuning is enabled
        if tune_kernel_size:
            model.block1.conv1.kernel_size = (kernel_size,)
            model.block2.conv1.kernel_size = (kernel_size,)
            model.block3.conv1.kernel_size = (kernel_size,)

        best_f1 = train_for_optuna(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=lr,
            weight_decay=weight_decay,
            p_drop=p_drop,
            class_weights=class_weights,
            train_one_epoch=train_one_epoch,
            evaluate=evaluate,
            device=device,
            max_epochs=max_epochs
        )
        print(f"Trial {trial.number} finished: Best F1={best_f1:.4f}\n")

        return best_f1
    
    return objective


def create_objective_recall(model_class, c_in, num_classes, device, train_loader, valid_loader,
                            class_weights, train_one_epoch, evaluate, max_epochs=15,
                            hidden_dim_options=[64, 128, 256], kernel_size_options=[3, 5, 7],
                            tune_kernel_size=False):
    """
    Create an objective function for Optuna optimization using recall.
    
    Args:
        model_class: The model class to instantiate (e.g., TinyTCN)
        c_in: Input channel dimension
        num_classes: Number of classes
        device: Device to run training on
        train_loader: Training data loader
        valid_loader: Validation data loader
        class_weights: Class weights for loss function
        train_one_epoch: Function to train one epoch
        evaluate: Function to evaluate the model
        max_epochs: Maximum number of epochs per trial
        hidden_dim_options: List of hidden dimension options to search
        kernel_size_options: List of kernel size options to search
        tune_kernel_size: Whether to tune kernel size (if False, kernel_size is ignored)
    
    Returns:
        objective: Function that can be passed to study.optimize()
    """
    def objective(trial):
        # Search space
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        p_drop = trial.suggest_float("p_drop", 0.0, 0.5)
        
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_options)
        kernel_size = trial.suggest_categorical("kernel_size", kernel_size_options)

        print(f"\n==============================")
        print(f"🔥 Starting Trial {trial.number}")
        print(f"Hyperparameters:")
        print(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}, p_drop={p_drop:.3f}")
        print(f"  hidden_dim={hidden_dim}, kernel_size={kernel_size}")
        print(f"==============================")

        model = model_class(c_in=c_in, num_classes=num_classes, p_drop=p_drop).to(device)

        # Optionally modify kernel size if tuning is enabled
        if tune_kernel_size:
            model.block1.conv1.kernel_size = (kernel_size,)
            model.block2.conv1.kernel_size = (kernel_size,)
            model.block3.conv1.kernel_size = (kernel_size,)

        best_recall = train_for_optuna_r(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=lr,
            weight_decay=weight_decay,
            p_drop=p_drop,
            class_weights=class_weights,
            train_one_epoch=train_one_epoch,
            evaluate=evaluate,
            device=device,
            max_epochs=max_epochs
        )
        print(f"Trial {trial.number} finished: Best Recall={best_recall:.4f}\n")

        return best_recall
    
    return objective


def run_hyperparameter_tuning(objective = "f1", n_trials=100, direction="maximize", study_name=None):
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        objective: Objective function (created by f1 or recall)
        n_trials: Number of trials to run
        direction: Direction of optimization ("maximize" or "minimize")
        study_name: Optional name for the study
    
    Returns:
        study: Optuna study object with results
    """
    objective_function = "create_objective_" + objective
    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)
    
    return study

