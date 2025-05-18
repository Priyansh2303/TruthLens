import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
test_dir = 'Dataset/Test'
model_path = 'model/mesonet_trained.h5'

# Load model
model = load_model(model_path)

# Prepare test data
img_height, img_width = 256, 256
batch_size = 16

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Predict
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# True labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Evaluation Report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))

# Confusion Matrix
print("\nConfusion Matrix:\n")
cm = confusion_matrix(true_classes, predicted_classes)

# Optional: Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
