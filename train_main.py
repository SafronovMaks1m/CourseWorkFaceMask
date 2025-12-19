import os
import cv2
import numpy as np
import joblib
import json
import tensorflow as tf
from skimage.feature import hog

from src.config import *
from src.data_loader import analyze_dataset
from src.models import create_classical_model, create_simple_cnn, create_transfer_model

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Adam = tf.keras.optimizers.Adam

def load_data_for_classical(directory):
    """Загрузка данных для Random Forest с извлечением HOG-признаков."""
    print(f"Загрузка данных для ML из {directory}...")
    features = []
    labels = []
    categories = ['WithMask', 'WithoutMask']

    for category in categories:
        path = os.path.join(directory, category)
        class_num = 1 if category == 'WithMask' else 0
        if not os.path.exists(path): continue
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue

                img_resized = cv2.resize(img, (HOG_IMG_SIZE, HOG_IMG_SIZE))
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                hog_vec = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
                
                features.append(hog_vec)
                labels.append(class_num)
            except:
                pass
    return np.array(features), np.array(labels)

def train_all():
    analyze_dataset()
    metrics = {} 

    print("\n--- Обучение Модели 1: HOG + Random Forest ---")
    X_train, y_train = load_data_for_classical(TRAIN_DIR)
    X_test, y_test = load_data_for_classical(TEST_DIR) 
    
    rf_model = create_classical_model()
    rf_model.fit(X_train, y_train)
    
    rf_acc = rf_model.score(X_test, y_test)
    metrics['rf_accuracy'] = float(rf_acc)
    print(f"Classical Model Test Accuracy: {rf_acc:.4f}")
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'classical_rf.pkl'))

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary')
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    print("\n--- Обучение Модели 2: Simple CNN ---")
    cnn = create_simple_cnn((IMG_SIZE, IMG_SIZE, 3))
    cnn.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(train_gen, epochs=EPOCHS)
    
    _, cnn_acc = cnn.evaluate(test_gen)
    metrics['cnn_accuracy'] = float(cnn_acc)
    cnn.save(os.path.join(MODELS_DIR, 'simple_cnn.keras'))

    print("\n--- Обучение Модели 3: MobileNetV2 ---")
    mobilenet = create_transfer_model((IMG_SIZE, IMG_SIZE, 3))
    mobilenet.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    mobilenet.fit(train_gen, epochs=EPOCHS)
    _, mob_acc = mobilenet.evaluate(test_gen)
    metrics['mobilenet_accuracy'] = float(mob_acc)
    mobilenet.save(os.path.join(MODELS_DIR, 'mobilenet.keras'))
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    print("\n✅ Все модели обучены. Метрики сохранены в models_saved/metrics.json")

if __name__ == "__main__":
    train_all()