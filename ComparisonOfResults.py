import joblib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载和预处理测试数据集
test_dir = 'DataSet/test'
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 加载ResNet50模型
resnet50_model = models.resnet50(weights=None)  # 不加载预训练权重
num_ftrs = resnet50_model.fc.in_features

# 定义与保存模型时相同的fc结构
resnet50_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)

# 加载ResNet50模型权重
resnet50_model.load_state_dict(torch.load('resnet50_ai_detection.pth'))
resnet50_model.eval()
resnet50_model.to(device)

# 3. 加载VGG19模型，用于特征提取
vgg19_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
# 保留VGG19的最后一个全连接层之前的所有层，以便提取4096维度的特征
vgg19_model.classifier = nn.Sequential(*list(vgg19_model.classifier.children())[:-1])
vgg19_model.eval()

# 加载One-Class SVM模型
svm_model = joblib.load('one_class_svm_model_pytorch.joblib')

# 4. 使用模型进行预测和评估
resnet50_predictions = []
svm_predictions = []
labels = []

with torch.no_grad():
    for images, label in test_loader:
        images = images.to(device)
        labels.extend(label.numpy())

        # ResNet50模型的预测
        resnet50_output = resnet50_model(images)
        _, resnet50_pred = torch.max(resnet50_output, 1)
        resnet50_predictions.extend(resnet50_pred.cpu().numpy())

        # VGG19提取特征，用于One-Class SVM
        vgg19_features = vgg19_model(images).cpu().numpy()  # 4096 维度特征
        svm_pred = svm_model.predict(vgg19_features)
        svm_predictions.extend(svm_pred)

# 5. 评估模型性能
accuracy_resnet50 = accuracy_score(labels, resnet50_predictions)
f1_resnet50 = f1_score(labels, resnet50_predictions, average='weighted')

accuracy_svm = accuracy_score(labels, svm_predictions)
f1_svm = f1_score(labels, svm_predictions, average='weighted')

# 6. 绘制对比图表并保存到本地
models = ['ResNet50', 'One-Class SVM']
accuracies = [accuracy_resnet50, accuracy_svm]
f1_scores = [f1_resnet50, f1_svm]

x = np.arange(len(models))  # 标签位置
width = 0.35  # 条形图的宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')

# 添加一些文本标签
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# 自动标注条形图的高度
def autolabel(rects):
    """在每个条形图顶部添加文本标签"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# 保存图表到本地
plt.savefig('model_comparison.png')

# 显示图表
plt.show()
