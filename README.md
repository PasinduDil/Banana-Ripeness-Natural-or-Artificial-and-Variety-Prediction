# 🍌 Banana Classification using VGG16

This project is a deep learning-based image classification model that classifies four types of bananas using a **pretrained VGG16** model. The model is trained on a dataset containing images of different banana types and achieves high accuracy through transfer learning.

---

## 📌 **Project Overview**
- **Objective**: Classify bananas into four categories:
  - `ambul_kesel_artificial`
  - `ambul_kesel_natural`
  - `Anamal_Natural`
  - `Anamalu_Artificial`
- **Model Used**: VGG16 (Pretrained on ImageNet)
- **Dataset**: Images of different banana types organized into separate folders.
- **Frameworks**: TensorFlow, Keras, NumPy, OpenCV, Matplotlib.
- **Training**: Model is trained using data augmentation and optimized with Adam optimizer.

---

## 📁 **Dataset Structure**
The dataset should be structured as follows:

/Dataset/
├── ambul_kesel_artificial/ # Images for this class
├── ambul_kesel_natural/ # Images for this class
├── Anamal_Natural/ # Images for this class
├── Anamalu_Artificial/ # Images for this class

Ensure each subfolder contains a **sufficient number of images** for effective training.

---

## 🚀 **Setup Instructions**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/yourusername/banana-classification.git
cd banana-classification

2️⃣ Install Dependencies
Install the required Python libraries:

pip install tensorflow keras numpy matplotlib opencv-python

📌 Future Improvements
Fine-tuning VGG16 layers to improve classification accuracy.
Collecting a larger dataset for better generalization.
Deploying the model using a web app (Flask/Streamlit) for real-time classification.
