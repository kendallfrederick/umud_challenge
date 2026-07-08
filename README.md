# umud_challenge
U-Net architecture for muscle architecture segmentation from ultrasound images

# Muscle Architecture Estimation from Ultrasound Images

## Project Overview

The goal of our project was to automatically estimate muscle architecture parameters from ultrasound images. Specifically, our pipeline estimates:

* **Muscle Thickness (MT)**
* **Pennation Angle (PA)**
* **Fascicle Length (FL)**

Rather than predicting these measurements directly, we first segmented the relevant anatomical structures (aponeuroses and fascicles) using deep learning. The resulting segmentation masks were then processed using classical image processing and geometric analysis to calculate the final measurements.

This approach makes the system more interpretable because every prediction can be visually inspected before the final measurements are computed.

---

# Overall Pipeline

The complete pipeline consists of the following stages:

1. Load and preprocess ultrasound images.
2. Train a U-Net to segment the aponeuroses.
3. Train a second U-Net to segment the fascicles.
4. Clean the predicted masks using morphological operations.
5. Extract the anatomical structures from the cleaned masks.
6. Compute:

   * Muscle Thickness (MT)
   * Pennation Angle (PA)
   * Fascicle Length (FL)
7. Save the final predictions to a CSV file.

---

# Design Decisions

## 1. Formulating the Problem as Image Segmentation

Rather than predicting MT, PA, and FL directly, we formulated the task as an image segmentation problem.

We trained two independent neural networks:

* One to segment the **aponeuroses**
* One to segment the **fascicles**

The predicted masks were then converted into anatomical measurements using geometric calculations.

### Why

The competition dataset already provides segmentation masks for training.

By segmenting the anatomical structures first, we preserved the spatial information required to compute accurate geometric measurements. This also made the pipeline significantly more interpretable because the intermediate segmentation masks can be visualized and inspected before any measurements are calculated.

### Alternative Approaches

One alternative would have been training a regression network to predict MT, PA, and FL directly from the ultrasound image.

We decided against this approach because:

* It provides no explanation for how the measurements were obtained.
* Incorrect predictions are difficult to debug.
* The available segmentation masks naturally support a segmentation-based solution.
* The available training dataset did not have labels for MT, PA and FL, making it hard to predict these directly without using binary masks.

---

## 2. Choosing U-Net

Both segmentation models use the U-Net architecture, one for segmenting the aponeuroses and another for segmenting the fascicles.

Our implementation consists of four main components:

- **Encoder**
- **Bottleneck**
- **Decoder**
- **Skip connections**

The network takes a **512 × 512 grayscale ultrasound image** as input.

The **encoder** contains three downsampling stages. Each stage consists of two 3×3 convolutional layers with ReLU activation followed by a max-pooling layer. The convolutional layers learn increasingly complex image features, while max pooling reduces the spatial resolution and allows the network to capture more abstract information. The number of filters increases from **32 → 64 → 128**.

The **bottleneck** is the deepest part of the network and contains two convolutional layers with **256 filters**. At this stage, the network has learned the highest-level representation of the image.

The **decoder** reconstructs the segmentation mask by progressively upsampling the feature maps back to the original image resolution. After each upsampling step, the feature maps are concatenated with the corresponding encoder features through **skip connections**, and two additional convolutional layers are applied. The number of filters decreases from **128 → 64 → 32**.

Finally, a **1×1 convolution with a sigmoid activation** produces a single-channel output where each pixel represents the probability of belonging to the target structure.

### Why We Chose U-Net

We chose U-Net because it is one of the most widely used architectures for biomedical image segmentation. The encoder captures both local and high-level image features, while the decoder reconstructs detailed segmentation masks. The skip connections help preserve fine spatial information that would otherwise be lost during downsampling, making U-Net particularly well suited for segmenting thin anatomical structures such as aponeuroses and fascicles.

---

## 3. Resizing Images to 512×512

All ultrasound images and masks were resized to **512 × 512 pixels** before training. Neural networks require images with consistent input dimensions.

Choosing a resolution of 512 × 512 provided a good compromise between:

* Preserving fine anatomical details.
* Maintaining reasonable GPU memory usage.
* Keeping training times manageable.

After prediction, all measurements were converted back into millimetres using the image metadata provided by the competition.

### Alternative Approaches

#### Original Image Sizes

Training on the original image sizes would preserve all image detail, but TensorFlow batches require fixed input dimensions, making training slower and significantly more memory intensive.

#### 256 × 256

This would reduce computational cost but would also remove important details, especially for thin fascicle structures.

#### 1024 × 1024

A higher resolution could preserve more anatomical detail but would substantially increase memory usage and training time.

---

## 4. Training and Validation Split

The dataset was divided into:

* 80% training
* 20% validation

using a fixed random seed.

---

## 5. Optimizer Selection

Both models were trained using the **Adam** optimizer. Adam automatically adapts the learning rate during training and generally performs well without extensive hyperparameter tuning.

Since this project had limited development time, Adam provided an efficient and reliable optimization method.

### Alternative Approaches

#### Stochastic Gradient Descent (SGD)

SGD can achieve excellent performance but usually requires careful tuning of the learning rate and momentum.

---

## 6. Binary Cross Entropy Loss

### What We Did

Our initial segmentation model was trained using Binary Cross Entropy (BCE) where each pixel belongs to one of two classes:

* Foreground
* Background

Binary Cross Entropy is therefore the standard loss function for binary segmentation.

### Limitation

Most pixels belong to the background.

This class imbalance means a model can achieve very high pixel accuracy while still producing poor segmentation masks.

---

## 7. Dice Loss

Our final loss function combined Binary Cross Entropy and Dice Loss:

```
Loss = BCE + Dice Loss
```

Dice Loss directly measures the overlap between the predicted segmentation mask and the ground truth.

Combining the two losses provides:

* Stable optimization from Binary Cross Entropy.
* Better overlap optimization from Dice Loss.

This generally produces more accurate segmentation masks than either loss alone.

---

## 8. Data Augmentation

### What We Did

During training we applied:

* Horizontal flips
* Small rotations
* Random zoom

The exact same transformation was applied to both the ultrasound image and its corresponding segmentation mask.

### Why

Medical datasets are typically small.

Data augmentation artificially increases dataset diversity, helping the model generalize better while reducing overfitting.

---

## 9. Early Stopping

Training automatically stopped if the validation loss failed to improve for five consecutive epochs.

The model weights corresponding to the best validation performance were restored.

---

## 10. Separate Networks for Aponeuroses and Fascicles

We trained two independent U-Net models:

* One specialized in aponeurosis segmentation.
* One specialized in fascicle segmentation.

These anatomical structures have very different visual characteristics.

Aponeuroses are relatively thick and continuous, whereas fascicles are much thinner and more fragmented.

Training separate models allowed each network to specialize in its own segmentation task.

### Alternative

A single multi-class segmentation model.

Although feasible, this would increase optimization complexity and could reduce segmentation quality for the thinner fascicle structures.

---

## 11. Classical Image Processing After Segmentation

### What We Did

After segmentation, we applied several traditional image processing techniques:

* Morphological cleaning
* Skeletonization
* Connected component analysis
* Line fitting
* Geometric calculations

These methods were then used to estimate:

* Muscle Thickness (MT)
* Pennation Angle (PA)
* Fascicle Length (FL)

These anatomical measurements are fundamentally geometric.

Using deterministic image processing after segmentation makes every measurement transparent, reproducible, and easy to verify visually.

# Model Performance

To evaluate the segmentation models, we measured both pixel-level and overlap-based metrics. While pixel accuracy provides an overall measure of correctly classified pixels, it can be misleading for highly imbalanced segmentation tasks where the background occupies most of the image. Therefore, we also report Precision, Recall, Dice Coefficient, and Intersection over Union (IoU), which better reflect the quality of the predicted segmentation masks.

| Metric | Aponeurosis Model | Fascicle Model |
|---------|------------------:|---------------:|
| Accuracy | **98.38%** | **99.42%** |
| Precision | **74.78%** | **18.47%** |
| Recall | **74.81%** | **28.21%** |
| Dice Coefficient | **0.748** | **0.223** |
| Intersection over Union (IoU) | **0.597** | **0.126** |

## Aponeurosis Model

The aponeurosis segmentation model achieved strong performance across all evaluation metrics. The Dice coefficient of **0.748** and IoU of **0.597** indicate good agreement between the predicted masks and the ground truth. Additionally, the precision and recall values are well balanced, showing that the model was able to correctly identify most aponeurosis pixels while producing relatively few false positives.

Overall, these results indicate that the model successfully learned to segment the superficial and deep aponeuroses, providing reliable inputs for the subsequent muscle architecture measurements.

## Fascicle Model

The fascicle segmentation model achieved a very high pixel accuracy (**99.42%**), but considerably lower overlap-based metrics, with a Dice coefficient of **0.223** and an IoU of **0.126**.

This difference is expected because fascicles occupy only a very small portion of each ultrasound image. As a result, correctly classifying the large background region leads to high overall accuracy even when the fascicle segmentation is imperfect. The lower precision and recall indicate that detecting the thin, fragmented fascicle structures is significantly more challenging than segmenting the larger and more continuous aponeuroses.

Despite the lower segmentation performance, the fascicle predictions remained useful after applying our post-processing pipeline. Skeletonization, connected-component analysis, filtering, and line fitting removed many spurious detections and allowed anatomically meaningful fascicle lines to be extracted for estimating pennation angle and fascicle length.


