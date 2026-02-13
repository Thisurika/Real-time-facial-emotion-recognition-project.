# Facial Emotion Recognition using Deep Learning

Real-time facial emotion detection system using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

Detects 7 emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**

## Demo

https://user-images.githubusercontent.com/xxxxxxxx/xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.mp4  
*(add your own short screen recording later – webcam demo)*

## Project Overview

This project implements a facial emotion recognition model using Keras/TensorFlow.  
The model was trained on grayscale 48×48 face images and can run in real-time using a webcam or on static images.

### Dataset

- **Name**: FER-2013 (Facial Expression Recognition 2013)
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/datasets/msambare/fer2013
- **Alternative (already in folder structure)**: https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013
- **Total images**: ≈ 35,887 grayscale 48×48 face images
- **Classes** (7 emotions):
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- **Class distribution**: Highly imbalanced (Happy ~8k images, Disgust ~500 images)

### Data Preprocessing

- Images are already 48×48 grayscale (no extra resizing needed)
- Normalization: pixel values divided by 255 → range [0, 1]
- Data augmentation applied during training:
  - Rotation (±30°)
  - Shear (0.3)
  - Zoom (0.3)
  - Horizontal flip
  - Fill mode: nearest

### Model Architecture

Simple but effective CNN:

- Input shape: (48, 48, 1)
- 4 convolutional blocks:
  - Conv2D (32 → 64 → 128 → 256 filters)
  - ReLU activation
  - MaxPooling2D (2×2)
  - Dropout (0.1)
- Flatten → Dense(512, ReLU) → Dropout(0.2) → Dense(7, softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Training: 30 epochs

## Folder Structure
Facial-Emotion-Project/

 ├── data/ 
 
 │   ├── train/
 
 │   │   ├── Angry/
 
 │   │   ├── Disgust/
 
 │   │   ├── Fear/
 
 │   │   ├── Happy/
 
 │   │   ├── Neutral/
 
 │   │   ├── Sad/
 
 │   │   └── Surprise/
 
 │   └── test/ 
 
 ├── tf-env/ 
 
 ├── main.py                      
 ├── test.py                      # real-time webcam demo
 
 ├── testdata.py                  # static image test
 
 ├── model_file.h5                # latest trained model (~29 MB)
 
 └── README.md

