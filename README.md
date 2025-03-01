# Neural Networks - Proof of Concept (PoC)

This repository presents a **Proof of Concept (PoC)** for using **Neural Networks (Classification Algorithm)** to recognize a specific face. The PoC includes a practical example:

- **Facial Recognition**: The model is trained to recognize **Al Pacino**, specifically in his role as **Tony Montana (Scarface)** to maintain consistency in age and facial features.

## 📌 **Project Objective**
This PoC demonstrates how **Convolutional Neural Networks (CNNs)** can be used for facial recognition. The goal is to classify images as **Al Pacino (Tony Montana)** or **not Al Pacino**.

## 🔍 **Implementation Details**
- **Dataset**: To keep the experiment simple, we used **only 50 images of Al Pacino** in the **Tony Montana (Scarface)** era.
- **Model**: A **Convolutional Neural Network (CNN)** with multiple layers to extract facial features.
- **Binary Classification**: The model outputs `1` for **Al Pacino (Tony Montana)** and `0` for **other faces**.
- **Training and Inference**: The system processes an uploaded image and determines if it matches the trained identity.

## ⚙ **Running the PoC**
### **1️⃣ Running in Google Colab** (Recommended)
This PoC was developed and tested on **Google Colab**, which provides free **GPU acceleration** for Machine Learning tasks. To run the model, simply:
1. Open the notebook in **Google Colab**.
2. Mount **Google Drive** (if necessary for model storage).
3. Run all cells to train or load the model.
4. Upload an image to test facial recognition.

### **2️⃣ Running on a Local Machine**
If running outside Google Colab, ensure your system has:
- **A compatible GPU** (NVIDIA recommended)
- **CUDA** (for GPU acceleration)
- **cuDNN** (for deep learning optimizations)
- **TensorFlow/Keras with GPU support**
- **Python 3.8+**

To install dependencies:
```sh
pip install tensorflow keras opencv-python matplotlib numpy scikit-learn
```

## 🚀 **Training and Using the Model**
The model can be trained or loaded from a saved version.

### **Train the Model (Optional)**
By default, the system **loads the pre-trained model**. However, if you want to re-train it, you must enter the password.

#### **Re-Training Password (Didactic Purpose Only)**
- The password **"foobar200"** allows re-training the model.
- This password is for **educational purposes only**.
- In a production system, a secure method like **.env files** should be used for sensitive information.

```sh
🔑 Enter password to re-train: foobar200
✅ Model is being re-trained...
```

### **Using a Pre-Trained Model**
If a trained model is found, the system **loads it automatically** instead of training from scratch.

```sh
🔄 No password entered. Loading the pre-trained model...
✅ Model loaded successfully!
```

## 📤 **Uploading an Image for Testing**
Once the model is loaded, the system enters an **infinite loop** until an image of **Al Pacino (Tony Montana)** is recognized.

1. **Upload an image** (JPG, PNG, WEBP).
2. The model predicts whether it is **Al Pacino (Tony Montana) or not**.
3. If recognized, the system stops.
4. Otherwise, it prompts for another upload.

```sh
📤 Upload an image to check recognition...
🔒 Not recognized. Try again.
🔓 Al Pacino recognized! Door opened and lights turned on.
✅ Access granted! System finished.
```

## 🌍 **Using the Model Without Google Drive**
If you don't want to mount Google Drive, you can **download the model from a public link** instead of re-training it.

### **Alternative: Downloading the Pre-Trained Model**
To avoid Google Drive dependency, host the trained model publicly (Google Drive, GitHub, Hugging Face, etc.), then modify the script to download it automatically.

#### **Example: Downloading from Google Drive**
```python
import urllib.request

# Public Google Drive link to model
model_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
model_filename = "al_pacino_model.keras"

# Download the model if not found
if not os.path.exists(model_filename):
    print("📥 Downloading pre-trained model...")
    urllib.request.urlretrieve(model_url, model_filename)
    print("✅ Model downloaded successfully!")

# Load the model
modelo_treinado = keras.models.load_model(model_filename)
print("✅ Model loaded and ready to use!")
```

## 📊 **Training Performance**
After training, a **performance graph** is generated showing:
- **Training Accuracy vs. Validation Accuracy**
- **Training Loss vs. Validation Loss**

This graph helps visualize model improvements and overfitting risks.

## 📜 **License**
This PoC is for educational and experimental purposes only.

---

🔗 **Developed for academic demonstration of Neural Networks (Classification Algorithm).** 🚀
By Mike Niner Bravog
