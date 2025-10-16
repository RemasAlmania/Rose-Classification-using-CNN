# Rose Classification using CNN

This project was developed as part of **CSC462: Machine Learning** at King Saud University.  
It implements a **Convolutional Neural Network (CNN)** for binary image classification to distinguish **roses** from other flowers.

---

## Project Description
The program:
1. Preprocesses the dataset and applies **data augmentation** (rotation, zoom, shifts, flips).
2. Splits the dataset into **training (70%)**, **validation (15%)**, and **test (15%)** sets.
3. Builds a **CNN model** with convolution, pooling, dropout, and batch normalization layers.
4. Trains the model with **early stopping** to avoid overfitting.
5. Evaluates the model using:
   - Precision, Recall, F1-score, Accuracy
   - Confusion matrices
   - ROC curves
   - Loss and accuracy curves

---

## Tools & Requirements
- **Python 3.9+**
- **TensorFlow / Keras**
- **scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

---

##  Dataset
We used a subset of the [Flowers Dataset on Kaggle](https://www.kaggle.com/datasets/imsparsh/flowers-dataset).  
From the original dataset, we selected:
- 100 images of roses
- 25 images of tulips
- 25 images of dandelions
- 25 images of sunflowers
- 25 images of daisies

We also applied data augmentation (rotation, zoom, shifting), which doubled the dataset size.

---
## Model Design
The custom CNN architecture includes:
- **3 Convolutional Layers** (filters: 16, 32, 64)
- **ReLU Activation**
- **2×2 MaxPooling** after each conv layer
- **Batch Normalization** and **Dropout (0.3)** for stability
- **L2 Regularization (0.001–0.005)** to prevent overfitting
- **Dense Layer (128 neurons)**
- **Output Layer:** Sigmoid (binary classification)

![5D8BB95F-232C-4217-B5C8-5BE0B193BAC2_1_201_a](https://github.com/user-attachments/assets/3e44bfd4-f42b-40dc-a261-ffa950509ed7)


---

## Model Development & Training

- Framework: TensorFlow & Keras
- Learning Rate: 0.0001
- Batch Size: 32 (training), 64 (validation/testing)
- Epochs: variable, with EarlyStopping (patience=5)
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
- Hyperparameter tuning for dropout, L2, and threshold adjustment


___

## Results

| Model                      | Accuracy | Precision | Recall | F1-Score |
| -------------------------- | -------- | --------- | ------ | -------- |
| **Custom CNN**             | 0.95     | 0.97      | 0.93   | 0.95     |
| **MobileNetV2 (Baseline)** | 0.92     | 0.93      | 0.90   | 0.92     |



![1F5CAFA8-2866-40A0-B4B4-797464398A34_1_201_a](https://github.com/user-attachments/assets/fcb97653-5578-47df-b721-f200dc6854d2)


---

##  Developed by
This project was developed by :
- [Remas Almania](https://github.com/RemasAlmania)
- [Hissah](https://github.com/hessakhs) 
- [Reem Alsuhaim](https://github.com/Reem-Alsuhaim)

--- 
## References 

1. Peryanto, A., Yudhana, A., & Umar, R. (2022). Convolutional neural network and support vector machine in classification of flower images. *Khazanah Informatika*.
2. Gurnani, A. et al. (2017). Flower Categorization using Deep Convolutional Neural Networks. *arXiv preprint*.
3. Alipour, N. et al. (2021). Flower Image Classification Using Deep Convolutional Neural Network. *IEEE ICWR*.
4. Howard, A. et al. (2017). Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv preprint*.


