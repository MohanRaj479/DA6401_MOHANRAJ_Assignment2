"""Unified multi-task model"""
import torch
import torch.nn as nn
import os
from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


  
