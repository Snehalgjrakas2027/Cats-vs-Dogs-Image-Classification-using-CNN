# 🐱🐶 Cats vs Dogs Image Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify images as either **cats** or **dogs**. It's a binary image classification task leveraging deep learning techniques to automatically learn visual features from image data.

---

## 🎯 Objective

To build a CNN-based model that can accurately distinguish between images of cats and dogs, helping in automating tasks such as image sorting, pet identification, and more.

---

## 🧾 Dataset

📁 **Dataset Used**: [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

* 25,000 labeled images of cats and dogs (JPEG format)
* Images are of various sizes, labeled as either `cat` or `dog` in filenames.

---

## 🛠️ Tools & Technologies

* **Python**
* **TensorFlow / Keras**
* **NumPy, Matplotlib, Pandas**
* **OpenCV / PIL** (for image preprocessing)

---

## 📊 Project Workflow

### 1. Data Preparation

* Load image data from directories
* Resize images to a consistent size (e.g., 128x128)
* Normalize pixel values (0–1)
* Split into training, validation, and test sets
* Use `ImageDataGenerator` for augmentation (rotation, flipping, etc.)

### 2. CNN Model Architecture

A typical CNN model includes:

* Convolutional layers with ReLU activation
* MaxPooling layers to reduce spatial dimensions
* Dropout layers to prevent overfitting
* Fully connected (Dense) layers
* Output layer with sigmoid activation (binary classification)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 3. Model Compilation & Training

* Loss: `binary_crossentropy`
* Optimizer: `adam`
* Metrics: `accuracy`
* Train for \~10–30 epochs with early stopping

### 4. Evaluation

* Evaluate model on test data
* Display classification accuracy
* Confusion matrix
* Sample predictions

---

## 📈 Results

| Metric         | Value  |
| -------------- | ------ |
| Accuracy       | \~95%+ |
| Loss           | Low    |
| Inference Time | Fast   |

---

## 🖼️ Sample Output

| Input Image              | Predicted Label |
| ------------------------ | --------------- |
| ![cat1](samples/cat.jpg) | Cat             |
| ![dog1](samples/dog.jpg) | Dog             |

---

## 🔧 Future Improvements

* Use pre-trained models (e.g., VGG16, ResNet50) for better accuracy
* Deploy model using Streamlit or Flask
* Convert model to TensorFlow Lite for mobile inference

