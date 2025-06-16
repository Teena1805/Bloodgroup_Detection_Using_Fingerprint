import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from enhance_fingerprint import enhance_fingerprint  # Make sure this file exists
from tensorflow.keras.applications.vgg16 import preprocess_input

# Paths
MODEL_PATH = 'Model/keras_model.h5'
LABELS_PATH = 'Model/labels.txt'
TEST_DIR = 'dataset_blood_group'  # Folder with actual test fingerprint images

# Load trained model
model = load_model(MODEL_PATH)

# Load class labels
label_map = {}
with open(LABELS_PATH, 'r') as f:
    for line in f:
        index, label = line.strip().split()
        label_map[int(index)] = label

# Ensure label ordering by index
label_list = [label_map[i] for i in sorted(label_map.keys())]

# Image parameters
IMAGE_SIZE = (224, 224)

# Initialize counters
total = 0
correct = 0

print("🧪 Evaluating test images...\n")

# Loop through each class folder
for class_name in os.listdir(TEST_DIR):
    class_folder = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)

        # Read grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Unable to read image: {image_path}")
            continue

        # Enhance fingerprint
        enhanced = enhance_fingerprint(image)

        # Resize and prepare for model
        resized = cv2.resize(enhanced, IMAGE_SIZE)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        x = img_to_array(rgb_image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict
        prediction = model.predict(x)
        predicted_index = np.argmax(prediction)
        predicted_label = label_list[predicted_index]

        total += 1
        if predicted_label == class_name:
            correct += 1
        else:
            print(f"❌ Mismatch: {image_name} | Actual: {class_name}, Predicted: {predicted_label}")

# Accuracy result
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
else:
    print("⚠️ No test images found in test_data folder.")
