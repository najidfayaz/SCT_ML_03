import sys
import os

# --- MAIN CODE ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Configuration
TRAIN_DIR = 'train\\train'  # The folder with the images
IMG_SIZE = 64        # Resize images to 64x64, faster processing
SAMPLE_SIZE = 2000   # Total images to use (1000 cats, 1000 dogs)

print(f"Loading {SAMPLE_SIZE} images from '{TRAIN_DIR}'...")

data = []
labels = []

# Load Images
cats = 0
dogs = 0

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: The folder '{TRAIN_DIR}' was not found.")
    sys.exit()

for img in os.listdir(TRAIN_DIR):
    # Check format
    if not img.endswith('.jpg'):
        continue
        
    # Limit no. of images to avoid crashing memory
    if cats >= SAMPLE_SIZE // 2 and dogs >= SAMPLE_SIZE // 2:
        break
        
    # Get Label 0 for Cat, 1 for Dog based on filename
    if 'cat' in img and cats < SAMPLE_SIZE // 2:
        label = 0
        cats += 1
    elif 'dog' in img and dogs < SAMPLE_SIZE // 2:
        label = 1
        dogs += 1
    else:
        continue

    # Read and Process Image
    path = os.path.join(TRAIN_DIR, img)
    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Convert to grayscale
    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE)) # Resize

    data.append(img_data.flatten()) # Flatten 2D to 1D array
    labels.append(label)

print(f"Loaded {len(data)} images. (Cats: {cats}, Dogs: {dogs})")

# Convert to Numpy Arrays
X = np.array(data)
X = X / 255.0  # Normalize pixel values (0-1)
y = np.array(labels)

# Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
print("Training SVM Model... might take a minute.")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Evaluate
print("Predicting...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Model Accuracy: {accuracy*100:.2f}% ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# Visualization of Predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    idx = np.random.randint(0, len(X_test))
    image = X_test[idx].reshape(IMG_SIZE, IMG_SIZE)
    prediction = "Dog" if y_pred[idx] == 1 else "Cat"
    actual = "Dog" if y_test[idx] == 1 else "Cat"
    
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Pred: {prediction}\nActual: {actual}")
    ax.axis('off')

plt.show()