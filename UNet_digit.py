import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_data(batch_size=800, image_size=64):
    """使用canvas方法确保数字居中"""
    images, masks, digits = [], [], []
    
    for _ in range(batch_size):
        digit = np.random.randint(0, 10)
        digits.append(digit)
        
        canvas_size = image_size * 2
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        font_scale = np.random.uniform(1.5, 2.0)
        thickness = np.random.randint(2, 4)
        text = str(digit)
        
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        center_x = canvas_size // 2
        center_y = canvas_size // 2
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2
        
        cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1.0, thickness)
        
        start_x = center_x - image_size // 2
        start_y = center_y - image_size // 2
        mask = canvas[start_y:start_y+image_size, start_x:start_x+image_size].copy()
        
        input_img = np.copy(mask) * np.random.uniform(0.7, 0.9)
        noise = np.random.randn(image_size, image_size) * np.random.uniform(0.05, 0.15)
        input_img = np.clip(input_img + noise, 0, 1)
        
        images.append(input_img)
        masks.append(mask)
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    images = np.expand_dims(images, axis=1)
    masks = np.expand_dims(masks, axis=1)
    
    return images, masks, digits

class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        dice_loss = 1 - dice_score
        return self.weight * bce_loss + (1 - self.weight) * dice_loss

def calculate_dice(pred, target):
    pred_binary = (pred > 0.5).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    return (2. * intersection + 1e-6) / (union + 1e-6)

class BalancedUNet(nn.Module):
    """平衡容量和效率的UNet"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m = self.middle(self.pool2(e2))
        
        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

def main():
    TOTAL_SIZE = 800
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 15
    TRAIN_RATIO = 0.8
    
    print(f"生成 {TOTAL_SIZE} 张 {IMAGE_SIZE}x{IMAGE_SIZE} 图像...")
    images, masks, digits = create_data(batch_size=TOTAL_SIZE, image_size=IMAGE_SIZE)
    
    train_idx, val_idx = train_test_split(
        np.arange(TOTAL_SIZE), 
        test_size=1-TRAIN_RATIO, 
        stratify=digits,
        random_state=42
    )
    
    train_images = images[train_idx]
    train_masks = masks[train_idx]
    val_images = images[val_idx]
    val_masks = masks[val_idx]
    
    print(f"训练集: {len(train_images)}张, 验证集: {len(val_images)}张")
    
    train_images_tensor = torch.FloatTensor(train_images)
    train_masks_tensor = torch.FloatTensor(train_masks)
    val_images_tensor = torch.FloatTensor(val_images)
    val_masks_tensor = torch.FloatTensor(val_masks)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = BalancedUNet().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    criterion = DiceBCELoss(weight=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, train_dices = [], []
    val_losses, val_dices = [], []
    best_val_dice = 0
    best_model_state = None
    
    print(f"开始训练，共 {EPOCHS} 个epoch...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss, epoch_train_dice = [], []
        
        indices = np.random.permutation(len(train_images))
        
        for i in range(0, len(train_images), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            batch_input = train_images_tensor[batch_idx].to(device)
            batch_target = train_masks_tensor[batch_idx].to(device)
            
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            dice = calculate_dice(output, batch_target).item()
            epoch_train_loss.append(loss.item())
            epoch_train_dice.append(dice)
        
        model.eval()
        epoch_val_loss, epoch_val_dice = [], []
        
        with torch.no_grad():
            for i in range(0, len(val_images), BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, len(val_images))
                batch_input = val_images_tensor[i:batch_end].to(device)
                batch_target = val_masks_tensor[i:batch_end].to(device)
                
                output = model(batch_input)
                loss = criterion(output, batch_target)
                dice = calculate_dice(output, batch_target).item()
                
                epoch_val_loss.append(loss.item())
                epoch_val_dice.append(dice)
        
        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_dice = np.mean(epoch_train_dice)
        avg_val_loss = np.mean(epoch_val_loss)
        avg_val_dice = np.mean(epoch_val_dice)
        
        train_losses.append(avg_train_loss)
        train_dices.append(avg_train_dice)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
              f"训练Dice={avg_train_dice:.4f}, 验证Dice={avg_val_dice:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n训练完成! 最佳验证Dice: {best_val_dice:.4f}")
    
    visualize_results(
        model, val_images_tensor, val_masks_tensor, 
        train_losses, val_losses, train_dices, val_dices
    )
    
    return {
        'train_dices': train_dices,
        'val_dices': val_dices,
        'best_val_dice': best_val_dice,
        'model_params': total_params
    }

def visualize_results(model, val_images, val_masks,
                     train_losses, val_losses, train_dices, val_dices):
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].plot(train_losses, label='训练损失', linewidth=2, color='blue')
    axes[0, 0].plot(val_losses, label='验证损失', linewidth=2, color='red')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(train_dices, label='训练Dice', linewidth=2, color='green')
    axes[0, 1].plot(val_dices, label='验证Dice', linewidth=2, color='orange')
    axes[0, 1].set_title('训练和验证Dice系数')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice系数')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].axis('off')
    summary_text = f"最佳验证Dice: {max(val_dices):.4f}\n最终训练Dice: {train_dices[-1]:.4f}\n最终验证Dice: {val_dices[-1]:.4f}"
    axes[0, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                   verticalalignment='center', horizontalalignment='left')
    
    model.eval()
    with torch.no_grad():
        sample_indices = np.random.choice(len(val_images), 3, replace=False)
        
        for i, idx in enumerate(sample_indices):
            ax = axes[1, i]
            
            img = val_images[idx:idx+1].to(device)
            mask = val_masks[idx:idx+1].to(device)
            pred = model(img)
            
            img_np = img[0, 0].cpu().numpy()
            mask_np = mask[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            pred_binary = (pred_np > 0.5).astype(np.float32)
            
            dice_value = calculate_dice(pred, mask).item()
            
            overlay = np.stack([img_np]*3, axis=-1)
            overlay[pred_binary > 0, 0] = 1.0
            overlay[mask_np > 0, 1] = 1.0
            
            overlap = (pred_binary > 0) & (mask_np > 0)
            overlay[overlap, 0] = 1.0
            overlay[overlap, 1] = 1.0
            overlay[overlap, 2] = 0.0
            
            ax.imshow(overlay)
            ax.set_title(f'Dice: {dice_value:.3f}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    try:
        results = main()
        print(f"\n结果总结:")
        print(f"模型参数量: {results['model_params']:,}")
        print(f"最佳验证Dice: {results['best_val_dice']:.4f}")
        if results['best_val_dice'] > 0.999:
            print("✓ 性能优秀: 达到0.999+")
        elif results['best_val_dice'] > 0.99:
            print("✓ 性能良好: 达到0.99+")
        else:
            print("○ 性能尚可")
    except Exception as e:
        print(f"运行出错: {e}")

