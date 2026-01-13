import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据集准备 ====================
def letters_target_transform(y):
    """将letters子集的标签从1-26转换为0-25"""
    return y - 1

def prepare_emnist_datasets(split='letters', data_root='./data', use_subset=True, subset_fraction=0.2):
    """准备EMNIST数据集，可选择使用子集"""
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    target_transform = letters_target_transform if split == 'letters' else None

    # 加载完整数据集
    train_dataset = datasets.EMNIST(
        root=data_root,
        split=split,
        train=True,
        download=True,
        transform=train_transform,
        target_transform=target_transform
    )
    test_dataset = datasets.EMNIST(
        root=data_root,
        split=split,
        train=False,
        download=True,
        transform=test_transform,
        target_transform=target_transform
    )

    class_names = [chr(ord('A') + i) for i in range(26)] if split == 'letters' else [str(i) for i in range(10)]
    
    if use_subset:
        # 使用部分训练数据加速
        subset_size = int(len(train_dataset) * subset_fraction)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)
        print(f"使用训练数据子集: {len(train_dataset)} (原始数据的{subset_fraction*100:.0f}%)")
    
    return train_dataset, test_dataset, class_names

# ==================== 2. ResNet18模型 ====================
class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)

class ResNet18ForEMNIST(nn.Module):
    """用于EMNIST的ResNet-18模型"""
    def __init__(self, num_classes=26):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层 - 针对28x28图像调整
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 四个残差阶段 [2, 2, 2, 2] 对应ResNet-18
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==================== 3. 训练函数（简化时间统计） ====================
def train_resnet_simple(model, train_loader, device, epochs=10):
    """简化版训练函数，只显示epoch总时间"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    train_losses, train_accs = [], []
    
    print("开始训练ResNet-18...")
    print("=" * 60)
    
    total_train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # 训练一个epoch
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 调整学习率
        scheduler.step()
        
        # 计算epoch总时间
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 只显示epoch总时间
        print(f'轮次 {epoch+1}/{epochs} | 损失: {epoch_loss:.4f} | 准确率: {epoch_acc:.2f}% | '
              f'时间: {epoch_time:.1f}秒')
    
    total_train_time = time.time() - total_train_start
    print(f"\n训练完成! 总训练时间: {total_train_time:.1f}秒 ({timedelta(seconds=int(total_train_time))})")
    return train_losses, train_accs, total_train_time

# ==================== 4. 评估函数 ====================
def evaluate_model(model, test_loader, device, class_names):
    """评估模型性能"""
    print("\n评估模型...")
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    accuracy = 100. * correct / total
    print(f"总体准确率: {correct}/{total} = {accuracy:.2f}%")
    
    # 显示最差和最好的5个类别
    class_accuracies = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            class_accuracies.append((class_names[i], class_acc))
    
    class_accuracies.sort(key=lambda x: x[1])
    print(f"\n最差的5个类别:")
    for i in range(min(5, len(class_accuracies))):
        print(f"  {class_accuracies[i][0]}: {class_accuracies[i][1]:.1f}%")
    
    print(f"\n最好的5个类别:")
    for i in range(1, min(6, len(class_accuracies)+1)):
        print(f"  {class_accuracies[-i][0]}: {class_accuracies[-i][1]:.1f}%")
    
    return accuracy

# ==================== 5. 可视化函数（已修复） ====================
def visualize_samples(dataset, class_names, num_samples=10):
    """可视化数据集样本 - 修复版"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    # 获取样本
    samples = []
    labels = []
    
    if isinstance(dataset, Subset):
        # 对于Subset数据集，我们需要从原始数据集中获取样本
        actual_dataset = dataset.dataset
        # 随机选择indices中的一些索引
        indices = np.random.choice(dataset.indices, min(num_samples, len(dataset)), replace=False)
        for idx in indices:
            img, label_idx = actual_dataset[idx]
            samples.append(img)
            labels.append(label_idx)
    else:
        # 对于普通数据集
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        for idx in indices:
            img, label_idx = dataset[idx]
            samples.append(img)
            labels.append(label_idx)
    
    # 显示所有样本
    for i, (img, label_idx) in enumerate(zip(samples, labels)):
        if i >= 10:  # 最多显示10个
            break
            
        row, col = i // 5, i % 5
        img_display = img[0].numpy()
        
        # 归一化显示
        img_min, img_max = img_display.min(), img_display.max()
        if img_max > img_min:
            img_display = (img_display - img_min) / (img_max - img_min)
        
        axes[row, col].imshow(img_display, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{class_names[label_idx]}')
        axes[row, col].axis('off')
    
    # 如果有空的子图，隐藏它们
    for i in range(len(samples), 10):
        row, col = i // 5, i % 5
        axes[row, col].axis('off')
    
    plt.suptitle('EMNIST 样本示例', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, train_accs):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-o', linewidth=2)
    axes[0].set_title('训练损失曲线', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('损失值')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(epochs, train_accs, 'g-s', linewidth=2)
    axes[1].set_title('训练准确率曲线', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('训练轮次')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.show()

# ==================== 6. 主执行流程 ====================
def main():
    """主执行函数"""
    program_start = time.time()
    
    print("=" * 70)
    print("EMNIST 字母识别 - ResNet-18 快速训练版")
    print("=" * 70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据集（使用30%数据加速）
    print("\n[1] 加载数据集...")
    train_dataset, test_dataset, class_names = prepare_emnist_datasets(
        split='letters', 
        use_subset=True, 
        subset_fraction=0.3 
    )
    
    print(f"训练数据: {len(train_dataset)}")
    print(f"测试数据: {len(test_dataset)}")
    print(f"字母类别: A-Z ({len(class_names)}个)")
    
    # 可视化样本
    print("\n[1.5] 可视化样本...")
    visualize_samples(train_dataset, class_names)
    
    # 2. 创建数据加载器（增大batch_size加速）
    print("\n[2] 创建数据加载器...")
    batch_size = 128  # 增大batch_size加速训练
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"训练批次: {len(train_loader)} (batch_size={batch_size})")
    
    # 3. 初始化ResNet-18模型
    print("\n[3] 初始化ResNet-18模型...")
    model = ResNet18ForEMNIST(num_classes=len(class_names)).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 训练模型
    print("\n[4] 训练模型...")
    epochs = 10
    train_losses, train_accs, train_time = train_resnet_simple(
        model, train_loader, device, epochs=epochs
    )
    
    # 5. 绘制训练曲线
    print("\n[5] 绘制训练曲线...")
    plot_training_curves(train_losses, train_accs)
    
    # 6. 评估模型
    print("\n[6] 评估模型...")
    accuracy = evaluate_model(model, test_loader, device, class_names)
    
    # 7. 保存模型
    print("\n[7] 保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'class_names': class_names,
    }, 'resnet18_emnist.pth')
    print("模型已保存为 'resnet18_emnist.pth'")
    
    total_time = time.time() - program_start
    print(f"\n总运行时间: {total_time:.1f}秒 ({timedelta(seconds=int(total_time))})")
    
    return model, accuracy, total_time

# ==================== 7. 执行主程序 ====================
if __name__ == "__main__":
    print("正在启动...")
    model, accuracy, total_time = main()
    print(f"\n程序执行完成! 最终测试准确率: {accuracy:.2f}%")
    print(f"总计耗时: {timedelta(seconds=int(total_time))}")
    
    # 性能分析
    print("\n" + "=" * 70)
    print("性能分析:")
    print(f"1. ResNet-18 在 30% 数据上训练 10 个epoch")
    print(f"2. 每个epoch耗时: {total_time/6:.1f}秒 (平均)")
    print(f"3. 最终准确率: {accuracy:.2f}%")
    print("=" * 70)

