import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import joblib

# 检查是否有可用的GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def preprocess_image(image, target_size=(224, 224)):
    intermediate_size = (256, 256)
    if image.size[0] > intermediate_size[0] or image.size[1] > intermediate_size[1]:
        image = image.resize(intermediate_size, Image.Resampling.LANCZOS)
    left = (image.width - target_size[0]) / 2
    top = (image.height - target_size[1]) / 2
    right = (image.width + target_size[0]) / 2
    bottom = (image.height + target_size[1]) / 2
    image = image.crop((left, top, right, bottom))
    return image

# 加载预训练的CNN模型
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def load_and_preprocess_data(dataset, batch_size=100):
    features = []
    batch_images = []
    for i, item in enumerate(dataset):
        image_data = item['image']
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        image = preprocess_image(image, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        batch_images.append(image)

        if len(batch_images) == batch_size:
            batch_images_np = np.array(batch_images)
            batch_features = model.predict(batch_images_np)
            for feature in batch_features:
                features.append(feature.flatten())
            batch_images = []
            print(f"Processed {i+1} images")

    # Process remaining images
    if batch_images:
        batch_images_np = np.array(batch_images)
        batch_features = model.predict(batch_images_np)
        for feature in batch_features:
            features.append(feature.flatten())
        print(f"Processed {i+1} images")

    return np.array(features)

# 加载和预处理数据
dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k", split='train', trust_remote_code=True)
print("Number of items before filtering:", len(dataset))

filtered_dataset = dataset.filter(lambda x: x['image_nsfw'] != 2)
print("Number of items after filtering:", len(filtered_dataset))

# Create a folder to save images
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

# 使用所有数据进行训练
features = load_and_preprocess_data(filtered_dataset, batch_size=100)

# 训练One-Class SVM模型
oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(features)

# 保存训练好的模型
model_filename = 'one_class_svm_model.joblib'
joblib.dump(oc_svm, model_filename)
print(f"Trained model saved as {model_filename}")
