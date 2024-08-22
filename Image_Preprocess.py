import os
import logging
import shutil
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np
from collections import Counter

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 确保ImageNetDataSet目录结构正确
if not os.path.exists('ImageNetDataSet/class0'):
    os.makedirs('ImageNetDataSet/class0', exist_ok=True)
    for filename in os.listdir('ImageNetDataSet'):
        if filename.endswith('.jpg'):
            shutil.move(os.path.join('ImageNetDataSet', filename), os.path.join('ImageNetDataSet/class0', filename))

# 加载数据集A
dataset_A = datasets.ImageFolder(root='ImageNetDataSet')
logger.info(f'Dataset A loaded with {len(dataset_A)} samples.')

# 加载并过滤数据集B
dataset_B = load_dataset("poloclub/diffusiondb", "2m_random_10k", split='train', trust_remote_code=True)
filtered_dataset_B = dataset_B.filter(lambda x: x['image_nsfw'] != 2)
logger.info(f'Dataset B loaded and filtered with {len(filtered_dataset_B)} samples.')

# 手动为数据集B中的所有样本添加标签为1
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.fromarray(np.array(item['image']))
        label = 1  # 手动添加标签为1
        return image, label

# 将过滤后的数据集B转换为CustomDataset
dataset_B = CustomDataset(filtered_dataset_B)

logger.info(f'Dataset B prepared with {len(dataset_B)} samples.')

# 检查数据集 A 的标签分布
labels_A = [label for _, label in dataset_A]
counter_A = Counter(labels_A)
logger.info(f'Dataset A label distribution: {counter_A}')

# 检查数据集 B 的标签分布
labels_B = [label for _, label in dataset_B]
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

def save_dataset_by_label(dataset, dataset_name):
    dataset_dir = os.path.join('DataSet', dataset_name)

    for i, (image, label) in enumerate(dataset):
        # 创建按标签划分的子文件夹
        label_dir = os.path.join(dataset_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # 保存图像
        image_path = os.path.join(label_dir, f'{i}.png')
        image.save(image_path)

    logger.info(f'{dataset_name} dataset saved to {dataset_dir}.')

# 保存训练集、验证集和测试集
save_dataset_by_label(train_dataset, 'train')
save_dataset_by_label(val_dataset, 'val')
save_dataset_by_label(test_dataset, 'test')
