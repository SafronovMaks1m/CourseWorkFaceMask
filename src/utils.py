import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image_hog(image, target_size=(64, 64)):
    """
    Подготовка изображения для классического метода (HOG).
    1. Ресайз.
    2. Перевод в оттенки серого.
    3. Извлечение HOG-вектора.
    """
    if image is None: return None
    img_resized = cv2.resize(image, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Извлечение признаков (Feature Extraction)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    return features.reshape(1, -1)

def preprocess_image_cnn(image, target_size=(128, 128)):
    """
    Подготовка изображения для нейросети.
    1. Ресайз.
    2. Нормализация (0-1).
    3. Добавление размерности (batch dimension).
    """
    if image is None: return None
    img_resized = cv2.resize(image, target_size)
    img_array = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)