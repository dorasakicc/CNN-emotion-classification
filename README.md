# Facial Emotion Recognition with Deep Learning

CNN-based emotion classification on FER2013 dataset using TensorFlow/Keras. Implemented and compared multiple deep learning architectures with advanced evaluation techniques.

**Course:** Arhitekture neuronskih mreza (Neural Network Architectures) — PMF Split

## Dataset

**FER2013** — 28,709 grayscale images of faces (48x48 px), 7 emotion classes:

| Emotion | Train Samples | Test Samples |
|---------|--------------|-------------|
| Happy | 7,215 | 1,774 |
| Sad | 4,830 | 1,247 |
| Neutral | 4,965 | 1,233 |
| Fear | 4,097 | 1,024 |
| Angry | 3,995 | 958 |
| Surprise | 3,171 | 831 |
| Disgust | 436 | 111 |

**Challenges:** Small image resolution (48x48), visually similar emotions (fear/sad/neutral), severe class imbalance (disgust has 16x fewer samples than happy).

## Results

| Approach | Test Accuracy |
|----------|:------------:|
| Transfer Learning (MobileNetV2) | ~42% |
| CNN (Baseline) | ~57% |
| CNN + TTA | ~60% |
| Ensemble (3 CNN) | ~59% |
| **Ensemble + TTA** | **~62%** |

**Key finding:** A custom CNN trained from scratch outperforms Transfer Learning (MobileNetV2) on this dataset. MobileNet features learned on ImageNet are not optimal for small grayscale face images.

## Architectures

### CNN (VGG-style, 4 blocks)
- 4 convolutional blocks with increasing filters: 32 → 64 → 128 → 256
- BatchNormalization + Dropout regularization in every block
- Adam optimizer with ReduceLROnPlateau scheduling

### Transfer Learning — MobileNetV2
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Images resized to 96x96 RGB
- Frozen base with custom classification head

### Convolutional Autoencoder
- Encoder: 48x48 → 6x6x8 latent space (8x compression)
- Decoder: reconstructs original image from latent representation
- MSE loss, trained separately from classifier

### Ensemble
- 3 independently trained CNN models with different weight initializations
- Averaged predictions reduce variance and improve robustness

## Techniques

| Technique | Purpose |
|-----------|---------|
| **Data Augmentation** | RandomFlip, RandomRotation, RandomZoom, RandomContrast |
| **Class Weights** | Upweight minority classes (disgust F1: 4% → 36%) |
| **Test Time Augmentation (TTA)** | Average 10 augmented predictions per image at inference |
| **K-Fold Cross Validation** | Validate model stability (std < 1%) |
| **Grad-CAM** | Visualize which facial regions CNN focuses on |
| **Feature Visualization** | Inspect learned filters and activation maps |

## Visualizations

### Grad-CAM — What the CNN looks at per emotion
Shows heatmaps of regions most important for classification (eyes, mouth, eyebrows).

### Feature Visualization
First convolutional layer learns edge detectors and texture patterns. Deeper layers combine these into complex facial structures.

### Autoencoder Reconstruction
Original vs reconstructed face images demonstrate the quality of learned latent representations.

## K-Fold Cross Validation

| Fold | Accuracy |
|------|:--------:|
| 1 | 57.73% |
| 2 | 57.76% |
| 3 | 58.22% |
| 4 | 57.05% |
| 5 | 58.21% |
| **Mean ± Std** | **57.79% ± 0.48%** |

Model is stable across different initializations — results are reproducible.

## How to Run

1. Open `neuronske_PROJEKT_FINALNI.ipynb` in Google Colab
2. Upload `fer_2013.zip` to your Google Drive
3. Run all cells — models are cached on Drive after first training
4. Subsequent runs load pretrained models automatically (~5 min instead of ~90 min)

## Tech Stack

- Python 3.12
- TensorFlow / Keras 2.20
- NumPy, Matplotlib, Seaborn
- scikit-learn
- Google Colab (T4 GPU)

## Future Work

- Audio stream (MFCC + LSTM) for multimodal emotion recognition
- Vision Transformer (ViT) architecture
- Larger datasets (AffectNet, RAF-DB) with institutional access
- Real-time emotion detection with webcam

## Author

**Dora Sakic** — PMF Split, Mathematics & Computer Science
