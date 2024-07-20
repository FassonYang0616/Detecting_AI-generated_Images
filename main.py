from datasets import load_dataset
import pandas as pd
from PIL import Image
import io
import os

# 从hugging face加载数据集
dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k", split='train', trust_remote_code=True)

# 输出加载的数据量
print("Number of items before filtering:", len(dataset))

# 过滤掉'image_nsfw'值为2的数据
filtered_dataset = dataset.filter(lambda x: x['image_nsfw'] != 2)

# 输出过滤后的数据量
print("Number of items after filtering:", len(filtered_dataset))

# 创建一个文件夹用于保存下载的数据集
image_folder = 'dataset_images'
os.makedirs(image_folder, exist_ok=True)

# 新建一个列表存储图片的文字数据
data_records = []

# 迭代过滤数据集中的每个项
for i, item in enumerate(filtered_dataset):
    # 提取图像数据，假设它在' image' 键中
    image_data = item['image']

    # 检查图像数据是否为字节格式并进行相应处理
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        # 如果图像数据已经是一个PIL image对象
        image = image_data
    # end if

    # 将这个image对象保存为.png格式
    image.save(f'{image_folder}/image_{i}.png')

    # 处理非图像的数据
    record = {key: val for key, val in item.items() if key != 'image'}
    record['image_path'] = f'{image_folder}/image_{i}.png'
    data_records.append(record)
# end for

# 将非图像数据存储为CSV文件
df = pd.DataFrame(data_records)
df.to_csv(f'{image_folder}/dataset_records.csv', index=False)

print("All data has been saved locally.")
