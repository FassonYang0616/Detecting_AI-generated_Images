import numpy as np
import cv2
from tensorflow.keras.optimizers import Sequence,ImageDataGenerator


class MomentumDifficultyBoostingDataGenerator(Sequence):
    def __init__(self, dataframe, x_col, y_col, batch_size, target_size, datagen_args, difficulty_momentum=0.9):
        self.dataframe = dataframe
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.difficulty_momentum = difficulty_momentum
        self.datagen = ImageDataGenerator(**datagen_args)
        self.difficulties = np.ones(len(dataframe))  # 初始化所有样本的难度评分为1
        self.indices = np.arange(len(dataframe))
        self.on_epoch_end()

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x, batch_y = self.__data_generation(indices)
        return batch_x, batch_y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        probabilities = self.difficulties / self.difficulties.sum()
        self.indices = np.random.choice(self.indices, size=len(self.indices), p=probabilities)

    def __data_generation(self, indices):
        batch_x = []
        batch_y = []
        for i in indices:
            img_path = self.dataframe.iloc[i][self.x_col]
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.target_size)
            img = self.datagen.random_transform(img)
            img = img / 255.0
            batch_x.append(img)
            batch_y.append(self.dataframe.iloc[i][self.y_col])
        return np.array(batch_x), np.array(batch_y)

    def update_difficulties(self, indices, losses):
        for i, loss in zip(indices, losses):
            self.difficulties[i] = self.difficulty_momentum * self.difficulties[i] + (
                        1 - self.difficulty_momentum) * loss


def augment_image(image):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)  # 水平翻转
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    scale = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image
