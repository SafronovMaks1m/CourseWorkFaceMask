import tensorflow as tf

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
MobileNetV2 = tf.keras.applications.MobileNetV2

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- 1. Классический метод (HOG + Random Forest) ---
# Преподаватель просил "Expert Decision Tree". RandomForest - это много деревьев.
def create_classical_model():
    # Используем Random Forest, так как он устойчив к выбросам
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42))
    return model

# --- 2. Простая CNN (Сверточная сеть) ---
def create_simple_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Защита от переобучения
        Dense(1, activation='sigmoid') # Выход 0 или 1 (бинарная классификация)
    ])
    return model

# --- 3. Transfer Learning (MobileNetV2) ---
def create_transfer_model(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Замораживаем веса базы
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model