import torch
import cv2
import os
import segmentation_models_pytorch as smp
import numpy as np

print("================ DIAGNOSTIC START ================")

# --- TEST 1: DATA LOADING ---
print("\n[TEST 1] Checking Data Loading...")
try:
    img_dir = "./dataset/train/images"
    if not os.path.exists(img_dir):
        print("FAIL: Dataset directory not found!")
    else:
        files = os.listdir(img_dir)
        if not files:
            print("FAIL: No images found in folder!")
        else:
            first_file = files[0]
            print(f"   Attempting to read: {first_file}")
            
            img_path = os.path.join(img_dir, first_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print("FAIL: cv2.imread returned None. Image might be corrupt.")
            else:
                print(f"   SUCCESS: Image loaded. Shape: {img.shape}")

except Exception as e:
    print(f"FAIL: Data error: {e}")

# --- TEST 2: GPU ARCHITECTURE ---
print("\n[TEST 2] Checking GPU & PyTorch...")
try:
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available. Using CPU?")
    else:
        device_name = torch.cuda.get_device_name(0)
        print(f"   GPU Detected: {device_name}")
        
        # Create a tiny dummy model
        print("   Initializing Model...")
        model = smp.Unet(encoder_name='resnet34', encoder_weights=None, classes=1)
        model.to('cuda')
        
        # Create fake data (Batch size 2)
        print("   Generating fake data (Ramdom Tensors)...")
        x = torch.rand(2, 3, 512, 512).to('cuda')
        
        print("   Running Forward Pass (This is usually where it crashes)...")
        y = model(x)
        print(f"   SUCCESS: Output shape {y.shape}")
        
        print("   Running Backward Pass...")
        loss = y.sum()
        loss.backward()
        print("   SUCCESS: Gradient calculation worked.")

except Exception as e:
    print(f"FAIL: GPU Error: {e}")
    print("\nSUGGESTION: This is likely a PyTorch version mismatch with the RTX 5070.")

print("\n================ DIAGNOSTIC END ================")