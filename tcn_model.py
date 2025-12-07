"""
TCN Model Definition

This module contains the Temporal Convolutional Network (TCN) model architecture
for figure skating jump rotation classification.

Usage Example:
    from tcn_model import TinyTCN
    
    model = TinyTCN(c_in=132, num_classes=2, p_drop=0.2)
    model = model.to(device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network Block with residual connection.
    
    Each block consists of:
    - Two 1D convolutions with batch normalization
    - Dropout for regularization
    - Residual connection (with projection if input/output channels differ)
    
    Args:
        c_in: Input channel dimension
        c_out: Output channel dimension
        k: Kernel size for convolutions
        dilation: Dilation rate for temporal convolution
        p_drop: Dropout probability
    """
    def __init__(self, c_in, c_out, k=5, dilation=1, p_drop=0.2):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.drop1 = nn.Dropout(p_drop)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, dilation=dilation, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.proj  = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through TCN block.
        
        Args:
            x: Input tensor of shape [B, C_in, T]
        
        Returns:
            Output tensor of shape [B, C_out, T]
        """
        res = self.proj(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + res, inplace=True)
        return x


class TinyTCN(nn.Module):
    """
    Tiny Temporal Convolutional Network for sequence classification.
    
    Architecture:
    - Three TCN blocks with increasing dilation rates (1, 2, 4)
    - Global Average Pooling and Global Max Pooling
    - Two fully connected layers for classification
    
    Args:
        c_in: Input channel dimension (e.g., 132 for pose keypoints)
        num_classes: Number of output classes
        p_drop: Dropout probability
    """
    def __init__(self, c_in=132, num_classes=2, p_drop=0.2):
        super().__init__()
        self.block1 = TCNBlock(c_in, 64,  k=5, dilation=1, p_drop=p_drop)
        self.block2 = TCNBlock(64,   64,  k=5, dilation=2, p_drop=p_drop)
        self.block3 = TCNBlock(64,   128, k=5, dilation=4, p_drop=p_drop)
        self.drop_head = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(128*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass through TinyTCN.
        
        Args:
            x: Input tensor of shape [B, C, T] where:
               B = batch size
               C = number of channels (c_in)
               T = temporal length
        
        Returns:
            logits: Output logits of shape [B, num_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        gap = x.mean(dim=-1)        # [B, 128] - Global Average Pooling
        gmp = x.amax(dim=-1)        # [B, 128] - Global Max Pooling
        h = torch.cat([gap, gmp], dim=1)  # [B, 256]
        h = self.drop_head(h)
        h = F.relu(self.fc1(h), inplace=True)
        return self.fc2(h)

