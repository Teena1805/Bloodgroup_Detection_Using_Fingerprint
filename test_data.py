import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"
TEST_DIR = "test_data"

# Load model and labels
model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip().split(maxsplit=1)[-1] for line in f.readlines()]

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# Evaluate
correct = 0
total = 0

for label in os.listdir(TEST_DIR):
    label_dir = os.path.join(TEST_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        try:
            input_tensor = preprocess_image(img_path)
            prediction = model.predict(input_tensor)[0]
            predicted_label = LABELS[np.argmax(prediction)]
            total += 1
            if predicted_label == label:
                correct += 1
        except Exception as e:
            print(f"Error with image {img_path}: {e}")

print(f"✅ Correct Predictions: {correct}")
print(f"❌ Incorrect Predictions: {total - correct}")
print(f"📊 Accuracy: {(correct / total) * 100:.2f}%")
