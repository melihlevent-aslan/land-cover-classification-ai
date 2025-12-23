import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
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

# Hyperparameters
BATCH_SIZE = 4   # Decrease to 8 if you run out of GPU memory
LEARNING_RATE = 0.0001
EPOCHS = 10

print(f"Using device: {DEVICE}")

# --- DATASET CLASS ---
class BuildingDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Add channel dimension and normalize
        mask = np.expand_dims(mask, axis=-1)
        mask = (mask > 0).astype('float32') # Binary mask (0 or 1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

# --- AUGMENTATION (Fixed Warnings) ---
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # Fixed: removed 'always_apply' which caused warnings
        albu.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=1), 
        albu.RandomCrop(height=512, width=512, p=1),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(512, 512, p=1)
    ]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    
    # 1. Initialize Model
    print("Initializing U-Net Model...")
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    model.to(DEVICE) # Move model to GPU
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # 2. Prepare DataLoaders
    x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'masks')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'masks')
    
    train_dataset = BuildingDataset(
        x_train_dir, y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    valid_dataset = BuildingDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # Note: num_workers=0 is safer on Windows to avoid multiprocessing errors
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Define Loss and Optimizer
    # New syntax: Access losses directly
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop (Manual)
    best_iou = 0.0
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") as pbar:
            for images, masks in pbar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(images)
                
                # Calculate loss
                loss = loss_fn(logits, masks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        # Simple IoU calculation variables
        intersection_total = 0
        union_total = 0
        
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                logits = model(images)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()
                
                # Calculate basic IoU for monitoring
                # Threshold at 0.5
                pred_mask = (logits > 0.5).float()
                intersection = (pred_mask * masks).sum()
                union = pred_mask.sum() + masks.sum() - intersection
                
                intersection_total += intersection
                union_total += union
        
        # Epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        iou_score = intersection_total / (union_total + 1e-7) # Add epsilon to avoid div by zero
        
        print(f"Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {iou_score:.4f}")
        
        # Save Best Model
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model, './best_model.pth')
            print(">>> New Best Model Saved! <<<")
            
    print("Training Complete.")