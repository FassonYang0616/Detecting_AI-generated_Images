import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageStat
import io
import cv2
import numpy as np
from datasets import load_dataset


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


def filter_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    stat = ImageStat.Stat(image)
    color_variance = sum(stat.var) / len(stat.var)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_count = np.sum(edges > 0)
    edge_threshold = 100
    color_threshold = 200

    if edge_count > edge_threshold or color_variance < color_threshold:
        return False
    return True


def load_and_preprocess_data():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k", split='train', trust_remote_code=True)
    print("Number of items before filtering:", len(dataset))

    filtered_dataset = dataset.filter(lambda x: x['image_nsfw'] != 2)
    filtered_dataset = filtered_dataset.filter(lambda x: filter_image(Image.open(io.BytesIO(x['image']))
                                                                      if isinstance(x['image'], bytes)
                                                                      else x['image']))
    print("Number of items after filtering:", len(filtered_dataset))

    custom_dataset = CustomDataset(filtered_dataset, transform=preprocess)
    return custom_dataset
