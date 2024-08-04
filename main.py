import random
import pandas as pd
from datasets import load_dataset
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Image_Preprocess import CustomDataset


def save_random_images(dataset, save_folder, num_images=10):
    os.makedirs(save_folder, exist_ok=True)
    random_indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(random_indices):
        image_data = dataset[idx]['image']
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = image_data

        image.save(os.path.join(save_folder, f'random_image_{i}.png'))
        plt.imshow(image)
        plt.show()


def load_and_preprocess_data():
    dataset = load_dataset("poloclub/diffusiondb", "2m_random_10k", split='train', trust_remote_code=True)
    print("Number of items before filtering:", len(dataset))

    # 过滤掉'image_nsfw'值为2的数据
    filtered_dataset = dataset.filter(lambda x: x['image_nsfw'] != 2)

    # 保存随机选择的10张图像到指定路径
    save_random_images(filtered_dataset, 'Dataset/DiffusionDB/DataSet')
    print("Number of items after filtering:", len(filtered_dataset))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    custom_dataset = CustomDataset(filtered_dataset, transform=preprocess)
    return custom_dataset


def save_preprocessed_data(dataset, image_folder, csv_path):
    # 验证文件夹是否存在，如果不存在则创建
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    data_records = []

    for i, (image, label) in enumerate(DataLoader(dataset, batch_size=1)):
        image_pil = transforms.ToPILImage()(image[0])
        image_path = os.path.join(image_folder, f'image_{i}.png')
        image_pil.save(image_path)

        record = {'image_path': image_path, 'label': label.item()}
        data_records.append(record)

    df = pd.DataFrame(data_records)
    df.to_csv(csv_path, index=False)
    print("All preprocessed data has been saved locally.")


def main():
    # 定义保存路径
    original_image_folder = 'Dataset/DiffusionDB/DataSet'
    preprocessed_image_folder = 'Dataset/DiffusionDB/AfterPreprocess'
    csv_path = os.path.join(preprocessed_image_folder, 'dataset_records.csv')
    # 加载和预处理数据
    custom_dataset = load_and_preprocess_data()

    # 保存预处理后的数据
    save_preprocessed_data(custom_dataset, preprocessed_image_folder, csv_path)


if __name__ == '__main__':
    main()
