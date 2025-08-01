from transformers import ViTConfig, ViTForImageClassification
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class DeiTSmall(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        model_name = 'facebook/deit-small-patch16-224'
        if pretrained:
            # Load pretrained model
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Initialize model from config without pretrained weights
            config = ViTConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
            )
            self.vit = ViTForImageClassification(config)

    def forward(self, x):
        # Handle any size input by using adaptive pooling if needed
        B, C, H, W = x.shape
        if H != 224 or W != 224:
            x = torch.nn.functional.interpolate(
                x,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )
        return self.vit(x).logits

def deit_small(num_classes=1000, pretrained=False):
    """
    Create ViT-Small model for Tiny ImageNet
    Args:
        num_classes: number of classes (default: 200 for Tiny ImageNet)
        pretrained: whether to load pretrained weights (default: True)
    """
    return DeiTSmall(num_classes, pretrained=pretrained)

class DeiTTiny(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        model_name = 'facebook/deit-tiny-patch16-224'
        if pretrained:
            # Load pretrained model
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Initialize model from config without pretrained weights
            config = ViTConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
            )
            self.vit = ViTForImageClassification(config)

    def forward(self, x):
        # Handle any size input by using adaptive pooling if needed
        B, C, H, W = x.shape
        if H != 224 or W != 224:
            x = torch.nn.functional.interpolate(
                x,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )
        return self.vit(x).logits

def deit_tiny(num_classes=1000, pretrained=False):
    """
    Create ViT-Small model for Tiny ImageNet
    Args:
        num_classes: number of classes (default: 200 for Tiny ImageNet)
        pretrained: whether to load pretrained weights (default: True)
    """
    return DeiTTiny(num_classes, pretrained=pretrained)
