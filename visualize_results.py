import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
MODEL_PATH = './best_model.pth'
VAL_IMG_DIR = './dataset/val/images'
VAL_MASK_DIR = './dataset/val/masks'
OUTPUT_DIR = './results_visualization'
SAMPLES_TO_TEST = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_visual_comparison(image_bgr, mask, prediction, filename):
    """
    Stitch 3 images together (Original | Truth | Prediction) using pure OpenCV
    """
    h, w, _ = image_bgr.shape
    
    # 1. Prepare Mask (Gray -> BGR for stacking)
    # Scale mask 0-1 to 0-255
    mask_vis = (mask * 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    
    # 2. Prepare Prediction (Binary -> Red Overlay)
    # Create a red overlay: B=0, G=0, R=255
    pred_vis = np.zeros_like(image_bgr)
    pred_vis[:, :, 2] = (prediction * 255).astype(np.uint8) # Red channel
    
    # Blend Prediction with Original Image (Weighted Add)
    # 70% Original + 30% Red Overlay
    pred_overlay = cv2.addWeighted(image_bgr, 0.7, pred_vis, 0.3, 0)
    
    # 3. Add Text Labels
    cv2.putText(image_bgr, "Satellite", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(mask_bgr, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(pred_overlay, "AI Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 4. Stitch Side-by-Side
    # Create a white separator line
    sep = np.ones((h, 10, 3), dtype=np.uint8) * 255
    combined = np.hstack((image_bgr, sep, mask_bgr, sep, pred_overlay))
    
    # 5. Save
    save_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
    cv2.imwrite(save_path, combined)
    print(f"   ---> Saved: {save_path}")

# --- EXECUTION ---
print("Starting robust visualization...")
print(f"Loading model from {MODEL_PATH}...")

# Load model safely
model = torch.load(MODEL_PATH, weights_only=False)
model = model.to(DEVICE)
model.eval()

# Get files
val_files = os.listdir(VAL_IMG_DIR)
np.random.shuffle(val_files)
selected_files = val_files[:SAMPLES_TO_TEST]

print(f"Generating {len(selected_files)} images...")

with torch.no_grad():
    for filename in selected_files:
        print(f"Processing: {filename}")
        
        # Load paths
        img_path = os.path.join(VAL_IMG_DIR, filename)
        mask_path = os.path.join(VAL_MASK_DIR, filename)
        
        # Load Images
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        
        mask = cv2.imread(mask_path, 0)
        if mask is None: continue
        
        # Prepare Input for AI
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = image_rgb.transpose(2, 0, 1).astype('float32')
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
        
        # Predict
        logits = model(input_tensor)
        pr_mask = torch.sigmoid(logits)
        pred_mask_np = pr_mask.squeeze().cpu().numpy()
        pred_binary = (pred_mask_np > 0.5).astype(np.uint8)
        
        # Save Visualization
        save_visual_comparison(image_bgr, mask, pred_binary, filename)

print("\nDone! Check the 'results_visualization' folder.")