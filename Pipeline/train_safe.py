import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = './dataset'
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAFE MODE SETTINGS
BATCH_SIZE = 4        # Low batch size for stability
LEARNING_RATE = 0.0001
EPOCHS = 5

print(f"Using device: {DEVICE}")

# --- ROBUST DATASET CLASS ---
class SafeBuildingDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
    
    def __getitem__(self, i):
        try:
            # 1. Read Image
            image = cv2.imread(self.images_fps[i])
            if image is None:
                raise ValueError(f"Image is None: {self.images_fps[i]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Read Mask
            mask = cv2.imread(self.masks_fps[i], 0)
            if mask is None:
                raise ValueError(f"Mask is None: {self.masks_fps[i]}")
            
            # 3. Preprocessing (Manual, removing library dependencies)
            # Expand mask dimensions
            mask = np.expand_dims(mask, axis=-1)
            # Normalize to 0-1
            mask = (mask > 0).astype('float32')
            
            # Convert to Tensor format (Channels First: HWC -> CHW)
            image = image.transpose(2, 0, 1).astype('float32')
            mask = mask.transpose(2, 0, 1).astype('float32')
            
            return image, mask

        except Exception as e:
            print(f"\n[WARNING] Skipping bad file at index {i}: {e}")
            # Return a dummy black image to keep training alive
            return torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512))
        
    def __len__(self):
        return len(self.ids)

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    
    print("Initializing Safe Mode U-Net...")
    
    # 1. Model
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    model.to(DEVICE)
    
    # 2. Data Loaders (No Augmentation)
    x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'masks')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'masks')
    
    train_dataset = SafeBuildingDataset(x_train_dir, y_train_dir)
    valid_dataset = SafeBuildingDataset(x_valid_dir, y_valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Loss & Opt
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Loop
    best_iou = 0.0
    print(f"Starting Safe Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        
        # TRAIN
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") as pbar:
            for images, masks in pbar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        # VALIDATION
        model.eval()
        intersection_total = 0
        union_total = 0
        
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                logits = model(images)
                pred_mask = (logits > 0.5).float()
                
                intersection = (pred_mask * masks).sum()
                union = pred_mask.sum() + masks.sum() - intersection
                intersection_total += intersection
                union_total += union
        
        iou_score = intersection_total / (union_total + 1e-7)
        print(f"Epoch {epoch+1} Result: Val IoU: {iou_score:.4f}")
        
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model, './best_model.pth')
            print(">>> Saved Best Model <<<")