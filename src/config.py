import os

# Пути к данным (относительно корня проекта)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

TRAIN_DIR = os.path.join(DATASET_DIR, 'Train')
VAL_DIR = os.path.join(DATASET_DIR, 'Validation')
TEST_DIR = os.path.join(DATASET_DIR, 'Test')

MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')
os.makedirs(MODELS_DIR, exist_ok=True)

# Параметры изображений
IMG_SIZE = 128      # Размер для нейросетей
HOG_IMG_SIZE = 64   # Размер для классического метода (меньше, чтобы быстрее считал)
BATCH_SIZE = 32
EPOCHS = 10         # Количество эпох обучения