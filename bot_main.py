import telebot
import numpy as np
import cv2
import joblib
import os
import tensorflow as tf

load_model = tf.keras.models.load_model

from src.config import MODELS_DIR
from src.utils import preprocess_image_hog, preprocess_image_cnn

API_TOKEN = '8594354830:AAGAKvM3P1oIDYMGJbl-j4lZJ7iyYwiHlrg'
bot = telebot.TeleBot(API_TOKEN)

print("Загрузка моделей...")
try:
    model_rf = joblib.load(os.path.join(MODELS_DIR, 'classical_rf.pkl'))
    model_cnn = load_model(os.path.join(MODELS_DIR, 'simple_cnn.keras'))
    model_mobile = load_model(os.path.join(MODELS_DIR, 'mobilenet.keras'))
    print("✅ Все модели успешно загружены.")
except Exception as e:
    print(f"❌ Ошибка загрузки моделей: {e}")
    print("Сначала запусти train_main.py!")
    exit()

def get_label_human(prob):
    if prob < 0.5:
        return "😷 В МАСКЕ", (1 - prob) * 100
    else:
        return "😐 БЕЗ МАСКИ", prob * 100

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне фото, и я проверю наличие маски тремя методами.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        np_arr = np.frombuffer(downloaded_file, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        response_text = "🔍 **Результаты анализа:**\n\n"

        hog_feat = preprocess_image_hog(img_rgb)
        rf_pred = model_rf.predict(hog_feat)[0]
        rf_label = "😷 В МАСКЕ" if rf_pred == 1 else "😐 БЕЗ МАСКИ"
        response_text += f"🔹 **Classical (HOG+RF):** {rf_label}\n"

        cnn_input = preprocess_image_cnn(img_rgb)
        cnn_prob = model_cnn.predict(cnn_input, verbose=0)[0][0]
        cnn_res, cnn_conf = get_label_human(cnn_prob)
        response_text += f"🔹 **Simple CNN:** {cnn_res} ({cnn_conf:.1f}%)\n"

        mob_prob = model_mobile.predict(cnn_input, verbose=0)[0][0]
        mob_res, mob_conf = get_label_human(mob_prob)
        response_text += f"🔹 **MobileNetV2:** {mob_res} ({mob_conf:.1f}%)\n"
        
        bot.reply_to(message, response_text, parse_mode="Markdown")
        
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")

print("Бот запущен...")
bot.polling()