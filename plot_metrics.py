import pickle
import matplotlib.pyplot as plt

# Load the saved history
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

acc = history.get('accuracy', [])
val_acc = history.get('val_accuracy', [])
loss = history.get('loss', [])
val_loss = history.get('val_loss', [])
precision = history.get('precision', [])
val_precision = history.get('val_precision', [])

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(16, 10))

# Accuracy Plot
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()

# Precision Plot
plt.subplot(2, 2, 3)
plt.plot(epochs_range, precision, label='Training Precision')
plt.plot(epochs_range, val_precision, label='Validation Precision')
plt.title('Precision')
plt.legend()

plt.tight_layout()
plt.show()
