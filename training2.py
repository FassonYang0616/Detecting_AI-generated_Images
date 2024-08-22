import os
import logging
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 定义图像预处理
class CustomTransform:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        img = self.transform(img)
        return img

# 目标尺寸
target_size = (224, 224)
custom_transform = CustomTransform(target_size)

# 确保ImageNetDataSet目录结构正确
if not os.path.exists('ImageNetDataSet/class0'):
    os.makedirs('ImageNetDataSet/class0', exist_ok=True)
    for filename in os.listdir('ImageNetDataSet'):
        if filename.endswith('.jpg'):
            shutil.move(os.path.join('ImageNetDataSet', filename), os.path.join('ImageNetDataSet/class0', filename))

# 加载数据集A
dataset_A = datasets.ImageFolder(root='ImageNetDataSet', transform=custom_transform)
logger.info(f'Dataset A loaded with {len(dataset_A)} samples.')

# 加载并过滤数据集B
dataset_B = load_dataset("poloclub/diffusiondb", "2m_random_10k", split='train', trust_remote_code=True)
filtered_dataset_B = dataset_B.filter(lambda x: x['image_nsfw'] != 2)
logger.info(f'Dataset B loaded and filtered with {len(filtered_dataset_B)} samples.')

# 自定义图像预处理函数
def preprocess_images(batch):
    images = [custom_transform(Image.fromarray(np.array(img)).convert('RGB')) for img in batch['image']]
    labels = torch.tensor([1] * len(images), dtype=torch.long)
    return {'image': images, 'label': labels}

# 预处理过滤后的数据集B
preprocessed_dataset_B = filtered_dataset_B.map(preprocess_images, batched=True, batch_size=32, remove_columns=['image'])

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        return image, label

# 将预处理后的数据集B转换为CustomDataset
dataset_B = CustomDataset(preprocessed_dataset_B)

logger.info(f'Dataset B preprocessed with {len(dataset_B)} samples.')

# 检查数据集 A 的标签分布
labels_A = [label for _, label in dataset_A]
counter_A = Counter(labels_A)
logger.info(f'Dataset A label distribution: {counter_A}')

# 检查数据集 B 的标签分布
labels_B = [label for label in preprocessed_dataset_B['label']]
counter_B = Counter(labels_B)
logger.info(f'Dataset B label distribution: {counter_B}')

# 划分数据集A和数据集B
def split_dataset(dataset, train_size, val_size, test_size):
    train_dataset, temp_dataset = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [val_size, test_size])
    return train_dataset, val_dataset, test_dataset

# 数据集划分比例
train_size_A = int(0.6 * len(dataset_A))
val_size_A = int(0.2 * len(dataset_A))
test_size_A = len(dataset_A) - train_size_A - val_size_A

train_size_B = int(0.6 * len(dataset_B))
val_size_B = int(0.2 * len(dataset_B))
test_size_B = len(dataset_B) - train_size_B - val_size_B

# 分别对数据集A和数据集B进行划分
train_A, val_A, test_A = split_dataset(dataset_A, train_size_A, val_size_A, test_size_A)
train_B, val_B, test_B = split_dataset(dataset_B, train_size_B, val_size_B, test_size_B)

# 合并训练集、验证集和测试集
train_dataset = torch.utils.data.ConcatDataset([train_A, train_B])
val_dataset = torch.utils.data.ConcatDataset([val_A, val_B])
test_dataset = torch.utils.data.ConcatDataset([test_A, test_B])

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack([torch.tensor(img) if isinstance(img, list) else img for img in images])
    labels = torch.tensor(labels)
    return images, labels

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

dataloaders = {'train': train_loader, 'val': val_loader}
logger.info('DataLoaders created.')

# 检查训练、验证和测试集的标签分布
def check_dataset_distribution(dataset, name):
    labels = [label for _, label in dataset]
    counter = Counter(labels)
    logger.info(f'{name} label distribution: {counter}')

check_dataset_distribution(train_dataset, 'Train Dataset')
check_dataset_distribution(val_dataset, 'Validation Dataset')
check_dataset_distribution(test_dataset, 'Test Dataset')

# 加载预训练的ResNet50模型
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 修改最后的全连接层并添加 Dropout 层
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)

# 设置损失函数和优化器，添加权重衰减
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 调整学习率和添加 L2 正则化

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f'Model loaded and moved to {device}.')

# 训练模型
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.cpu().numpy())

    return model, train_losses, val_losses, train_accuracies, val_accuracies

model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

# 绘制损失图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# 绘制准确率图
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# 测试模型
def test_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(dataloader.dataset)
    accuracy = running_corrects.double() / len(dataloader.dataset)

    logger.info(f'Test Loss: {loss:.4f} Acc: {accuracy:.4f}')
    return loss, accuracy

test_loss, test_accuracy = test_model(model, test_loader, criterion)

# 保存模型
torch.save(model.state_dict(), 'resnet50_ai_detection.pth')
logger.info('Model saved as resnet50_ai_detection.pth.')

# 评估模型性能
def predict(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

true_labels, predictions = predict(model, test_loader)

conf_matrix = confusion_matrix(true_labels, predictions)
logger.info('Confusion Matrix:')
logger.info(f'\n{conf_matrix}')

class_report = classification_report(true_labels, predictions, target_names=['Class 0', 'Class 1'], zero_division=0)
logger.info('Classification Report:')
logger.info(f'\n{class_report}')

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
