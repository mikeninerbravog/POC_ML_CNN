### **📌 Code Breakdown (English)**

This script is a **Proof of Concept (PoC)** that demonstrates how **Neural Networks (CNN - Convolutional Neural Networks)** can be used for **Facial Recognition**. The model is trained to recognize **Al Pacino** (specifically his role as **Tony Montana in Scarface**) and differentiate him from other faces. 

---

## **1️⃣ Importing Required Libraries**
The script begins by **importing essential libraries** for data processing, machine learning, and visualization.

```python
from google.colab import drive
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files
```

### **🔹 What Each Library Does**
- **TensorFlow/Keras**: Used for training the CNN model.
- **OpenCV (cv2)**: For image processing.
- **Matplotlib**: For visualizing results.
- **Scikit-learn (train_test_split)**: To split data into training and testing sets.
- **Google Colab Files**: Allows users to upload images for testing.

---

## **2️⃣ Mounting Google Drive**
Since the dataset and model are stored in **Google Drive**, the script **mounts Google Drive** to access files.

```python
drive.mount('/content/drive')
```

This step is **required** for accessing the dataset and saving/loading the trained model.

---

## **3️⃣ Setting Paths for Dataset and Model**
The script defines **paths** for the dataset and saved model.

```python
# Path to save the trained model
model_save_path = "/content/drive/MyDrive/al_pacino_model.keras"

# Path to the dataset (training images)
dataset_path = "/content/drive/MyDrive/dataset"
```

- The dataset is assumed to be stored in **Google Drive**.
- The model is saved in the **Keras format (`.keras`)**.

---

## **4️⃣ Image Preprocessing**
This function **loads and processes images** from a given folder.

```python
def carregar_imagens(diretorio, label):
    imagens = []
    labels = []
    for file in os.listdir(diretorio):
        if file.endswith((".jpg", ".png", ".webp")):
            img_path = os.path.join(diretorio, file)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip corrupted images
            img = cv2.resize(img, (128, 128))  # Resize to CNN input size
            img = img / 255.0  # Normalize pixel values (0-1)
            imagens.append(img)
            labels.append(label)  # 1 = Al Pacino, 0 = Other Faces
    return np.array(imagens), np.array(labels)
```

### **🔹 Key Features**
- **Reads images** from the directory.
- **Resizes** them to **128x128 pixels** (required by the CNN).
- **Normalizes pixel values** (scales between 0 and 1).
- **Assigns labels** (`1` for Al Pacino, `0` for others).

---

## **5️⃣ Model Training (Optional)**
By default, the script **loads a pre-trained model**. However, users can **re-train it** by entering the password.

```python
senha = input("🔑 Enter password to re-train the model (or press Enter to load the saved model): ")

if senha == "foobar200":
```

### **🔹 Training Process**
- **Loads images** from `al_pacino` and `outros/allImages`.
- **Splits data** into 80% training and 20% testing.
- **Uses Data Augmentation** to avoid overfitting.
- **Creates a CNN architecture** with **three convolutional layers**.
- **Uses `adam` optimizer** and **binary cross-entropy** as the loss function.
- **Trains for 50 epochs**, with early stopping after 3 non-improving epochs.

```python
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output (1 = Al Pacino, 0 = Other)
])
```

---

## **6️⃣ Saving and Loading the Model**
### **🔹 Saving the Model After Training**
Once trained, the model is saved in **Google Drive**.

```python
model.save(model_save_path)
print(f"✅ Model re-trained and saved at: {model_save_path}")
```

### **🔹 Loading a Pre-Trained Model**
If the password is **not provided**, the script loads the existing model instead.

```python
elif os.path.exists(model_save_path):
    print("🔄 Incorrect or empty password. Loading pre-trained model...")
    return keras.models.load_model(model_save_path)
```

If no model exists, it **forces training**.

```python
else:
    raise FileNotFoundError("❌ No saved model found and incorrect password. Training is required.")
```

---

## **7️⃣ Facial Recognition Function**
This function **predicts whether a given image is Al Pacino or not**.

```python
def prever_rosto(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False, "🔒 Image not found. Nothing happens."
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for CNN

    previsao = model.predict(img)[0][0]
    if previsao > 0.5:
        return True, "🔓 Al Pacino recognized! Door opened and lights turned on."
    else:
        return False, "🔒 Not recognized. Nothing happens."
```

### **🔹 Prediction Logic**
- Loads and preprocesses the image.
- Passes it through the CNN model.
- If **prediction > 0.5**, **Al Pacino is recognized**.
- Otherwise, the system **waits for a correct image**.

---

## **8️⃣ Image Upload and Loop Until Recognition**
The script enters a **loop** until the user uploads an image that the model recognizes as **Al Pacino**.

```python
def upload_e_prever(model):
    print("📤 Upload an image. The system will continue until Al Pacino is recognized!")

    while True:
        uploaded = files.upload()

        for file_name in uploaded.keys():
            image_path = file_name
            print(f"🖼 Image uploaded: {file_name}")
            
            reconhecido, resultado = prever_rosto(model, image_path)
            print(resultado)

            img = cv2.imread(image_path)
            if img is not None:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("Tested Image")
                plt.axis("off")
                plt.show()

            if reconhecido:
                print("✅ Access granted! System finished.")
                return

            print("⏳ Try again. The system will continue until Al Pacino is recognized.")
```

- **Continuously prompts the user** to upload images.
- **Only stops** when **Al Pacino (Tony Montana) is recognized**.

---

## **9️⃣ Running the System**
Finally, the script **trains (if needed) and starts the recognition process**.

```python
modelo_treinado = treinar_modelo()
upload_e_prever(modelo_treinado)
```

---

## **🔥 Key Features Recap**
✅ **Uses Convolutional Neural Networks (CNNs) for Facial Recognition**  
✅ **Recognizes Al Pacino as Tony Montana (Scarface)**  
✅ **Trains with only ~50 images for demonstration purposes**  
✅ **Allows re-training with password (didactic use only)**  
✅ **Loads pre-trained model if available**  
✅ **Works in an infinite loop until Al Pacino is detected**  

This breakdown explains each part of the **Facial Recognition PoC using Neural Networks**. 🚀 Let me know if you need modifications!
