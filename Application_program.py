import torch
from torchvision import transforms
from PIL import Image
import joblib
import io
import torchvision.models as models
from Image_Preprocess import filter_image  # 从 preprocessing.py 导入 filter_image

# 定义图像预处理步骤（与训练时一致）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的VGG19模型和训练好的One-Class SVM模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(pretrained=True).to(device)
model.classifier = model.classifier[:-1]  # 去掉最后一层
model.eval()
oc_svm = joblib.load('one_class_svm_model_pytorch.joblib')

# 用户端图像检测函数
def detect_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 应用过滤步骤
    if not filter_image(image):
        return "Image filtered out due to unrealistic style"

    # 图像预处理
    image = preprocess(image).unsqueeze(0).to(device)

    # 特征提取
    with torch.no_grad():
        feature = model(image).cpu().numpy()

    # 模型预测
    prediction = oc_svm.predict(feature)
    return 'Real' if prediction == 1 else 'AI Generated'

# 示例：用户上传图像
with open("path_to_uploaded_image", "rb") as f:
    image_bytes = f.read()

result = detect_image(image_bytes)
print(f"Prediction for uploaded image: {result}")
