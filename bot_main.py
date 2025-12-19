import telebot
from telebot import types
import numpy as np
import cv2
import joblib
import os
import json
import tensorflow as tf

from src.config import MODELS_DIR
from src.utils import preprocess_image_hog, preprocess_image_cnn

load_model = tf.keras.models.load_model
API_TOKEN = '8594354830:AAGAKvM3P1oIDYMGJbl-j4lZJ7iyYwiHlrg'
bot = telebot.TeleBot(API_TOKEN)
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')

try:
    model_rf = joblib.load(os.path.join(MODELS_DIR, 'classical_rf.pkl'))
    model_cnn = load_model(os.path.join(MODELS_DIR, 'simple_cnn.keras'))
    model_mobile = load_model(os.path.join(MODELS_DIR, 'mobilenet.keras'))
    print("✅ Модели загружены")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    exit()

def get_label_human(prob):
    if prob < 0.5:
        return "😷 В МАСКЕ", (1 - prob) * 100
    else:
        return "😐 БЕЗ МАСКИ", prob * 100

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton("📊 Показать точность моделей"))
    bot.reply_to(message, "Отправь фото или нажми на кнопку статистики.", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "📊 Показать точность моделей")
def handle_stats(message):
    if not os.path.exists(METRICS_PATH):
        bot.reply_to(message, "⚠️ Файл статистики не найден. Сначала обучи модели!")
        return

    with open(METRICS_PATH, 'r') as f:
        data = json.load(f)

    res = "📈 **Точность на папке Test:**\n\n"
    res += f"🔹 **HOG + RF:** {data.get('rf_accuracy', 0)*100:.2f}%\n"
    res += f"🔹 **Simple CNN:** {data.get('cnn_accuracy', 0)*100:.2f}%\n"
    res += f"🔹 **MobileNetV2:** {data.get('mobilenet_accuracy', 0)*100:.2f}%"
    bot.send_message(message.chat.id, res, parse_mode="Markdown")

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
        rf_probs = model_rf.predict_proba(hog_feat)[0]
        prob_mask_rf = rf_probs[1]
        
        if prob_mask_rf > 0.5:
            rf_label, rf_conf = "😷 В МАСКЕ", prob_mask_rf * 100
        else:
            rf_label, rf_conf = "😐 БЕЗ МАСКИ", (1 - prob_mask_rf) * 100
        response_text += f"🔹 **Classical (HOG+RF):** {rf_label} ({rf_conf:.1f}%)\n"

        cnn_input = preprocess_image_cnn(img_rgb)
        
        cnn_prob = model_cnn.predict(cnn_input, verbose=0)[0][0]
        cnn_res, cnn_conf = get_label_human(cnn_prob)
        response_text += f"🔹 **Simple CNN:** {cnn_res} ({cnn_conf:.1f}%)\n"

        mob_prob = model_mobile.predict(cnn_input, verbose=0)[0][0]
        mob_res, mob_conf = get_label_human(mob_prob)
        response_text += f"🔹 **MobileNetV2:** {mob_res} ({mob_conf:.1f}%)\n"
        
        bot.reply_to(message, response_text, parse_mode="Markdown")
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

if __name__ == "__main__":
    bot.polling(none_stop=True)