import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
import joblib
import numpy as np
from PIL import Image
from Image_Preprocess import load_and_preprocess_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
