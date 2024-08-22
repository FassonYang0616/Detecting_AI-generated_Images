import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
import joblib
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageStat
import io
import cv2
import numpy as np
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
    print("Number of items after filtering:", len(filtered_dataset))

    custom_dataset = CustomDataset(filtered_dataset, transform=preprocess)
    return custom_dataset

def extract_features(data_loader, model, device):
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
    return np.concatenate(features, axis=0)

custom_dataset = load_and_preprocess_data()
data_loader = DataLoader(custom_dataset, batch_size=100, shuffle=False)

model = models.vgg19(pretrained=True).to(device)
model.classifier = model.classifier[:-1]
model.eval()

all_features = []
for batch_idx, (images, _) in enumerate(data_loader):
    batch_features = extract_features(DataLoader(list(zip(images, _)), batch_size=len(images)), model, device)
    all_features.append(batch_features)
    print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

all_features = np.concatenate(all_features, axis=0)

oc_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(all_features)

model_filename = 'one_class_svm_model_pytorch.joblib'
joblib.dump(oc_svm, model_filename)
print(f"Trained model saved as {model_filename}")

def test_model(image, model, oc_svm, preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image).cpu().numpy()
    prediction = oc_svm.predict(feature)
    return prediction

test_image = Image.open("path_to_test_image").convert("RGB")
prediction = test_model(test_image, model, oc_svm, custom_dataset.transform)
print(f"Prediction for test image: {'Real' if prediction == 1 else 'AI Generated'}")
