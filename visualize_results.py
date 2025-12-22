import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
MODEL_PATH = './best_model.pth'
VAL_IMG_DIR = './dataset/val/images'
VAL_MASK_DIR = './dataset/val/masks'
OUTPUT_DIR = './results_visualization'
SAMPLES_TO_TEST = 10  # How many images to generate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'