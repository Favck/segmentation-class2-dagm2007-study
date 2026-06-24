# Industrial Defect Segmentation on DAGM 2007 (Class 2)

A PyTorch-based Deep Learning project focused on semantic segmentation for industrial quality control, using the **DAGM 2007 dataset (Class 2)**. This repository demonstrates an end-to-end pipeline using the U-Net architecture, partial weight fine-tuning, and a mathematically sound evaluation setup to detect anomalies and defects on textured surfaces.

## 📊 Final Performance (Test Set)

After fixing the soft-target evaluations and removing data augmentations from the test pipeline, the model achieves highly accurate and reproducible results on completely unseen data:

* **Test Jaccard Index (IoU Score):** **`0.6592`** (~65.9% strict pixel-level intersection)
* **Test Dice Loss:** **`0.2696`** (Translates to a **`0.7304`** / ~73.0% Dice Similarity Score)

---

## 🚀 Pipeline & Training Architecture

### 1. Model Configuration
* **Architecture:** U-Net from the `segmentation_models_pytorch` library.
* **Encoder:** **ResNet34** pre-trained on `imagenet`.
* **Input/Output Channels:** 1 channel (Grayscale input images and binary single-channel mask outputs).

### 2. Optimization & Partial Fine-Tuning
* **Optimizer:** **AdamW** with a learning rate of `3e-4` for smooth convergence without sharp gradient oscillations.
* **Weight Freezing:** To leverage pre-trained features and completely eliminate overfitting, the early feature-extracting layers of the encoder were frozen. Only the randomly initialized U-Net decoder blocks (`model.decoder`) were fully trained.

### 3. Loss Functions & Evaluation Metrics
* **Dice Loss:** Implemented as a custom `nn.Module` computing $1 - \text{Dice Score}$, ensuring that the optimizer correctly minimizes the training error toward zero instead of maximizing similarity blindly.
* **Jaccard Index:** Managed via `torchmetrics.JaccardIndex`. Evaluated strictly after passing raw model logits through a `torch.sigmoid()` activation to align with the binary thresholding logic (`threshold=0.5`).

---

## 📈 Training Progress & Convergence

Below is the visualization of the training phase. The model smoothly converges to a stable plateau around the 20th epoch, showing no signs of severe overfitting (validation error closely tracks the training loss).

<img width="855" height="559" alt="изображение" src="https://github.com/user-attachments/assets/aaab5cab-d29d-4dbd-937e-b7eb28dace98" />


---

## 🛠️ Project Structure

```text
├── dataset.py          # Custom PyTorch Dataset class handles image loading, resizing & transforms
├── train.py            # Complete training, validation, and testing loops
├── requirements.txt    # Set of required dependencies
└── README.md           # Project documentation and results presentation
