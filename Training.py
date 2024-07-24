import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import joblib
from datasets import load_dataset

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_data = self.dataset[idx]['image']
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = image_data
        label = self.dataset[idx]['image_nsfw']

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的CNN模型
model = models.vgg19(pretrained=True).to(device)
model.classifier = model.classifier[:-1]  # 去掉最后一层
model.eval()

# 加载数据集
dataset = load_dataset("poloclub/diffusiondb", "2m_random_100k", split='train', trust_remote_code=True)
print("Number of items before filtering:", len(dataset))
# 过滤数据
filtered_dataset = dataset.filter(lambda x: x['image_nsfw'] != 2)
print("Number of items after filtering:", len(filtered_dataset))
# 创建数据集和数据加载器
custom_dataset = CustomDataset(filtered_dataset, transform=preprocess)
data_loader = DataLoader(custom_dataset, batch_size=100, shuffle=False)

# 提取特征
def extract_features(data_loader, model, device):
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
    return np.concatenate(features, axis=0)

# 使用所有数据进行特征提取
features = extract_features(data_loader, model, device)

# 训练One-Class SVM模型
oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(features)

# 保存训练好的模型
model_filename = 'one_class_svm_model_pytorch.joblib'
joblib.dump(oc_svm, model_filename)
print(f"Trained model saved as {model_filename}")

# 保存数据记录和图像
image_folder = 'dataset_images'
os.makedirs(image_folder, exist_ok=True)
data_records = []

for i, item in enumerate(filtered_dataset):
    image_data = item['image']
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = image_data
    image.save(f'{image_folder}/image_{i}.png')
    record = {key: val for key, val in item.items() if key != 'image'}
    record['image_path'] = f'{image_folder}/image_{i}.png'
    data_records.append(record)

df = pd.DataFrame(data_records)
df.to_csv(f'{image_folder}/dataset_records.csv', index=False)

print("All data has been saved locally.")
