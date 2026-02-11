# ICSSR-CIFAR10-Image-Classification
Train an image classifier for the 10 CIFAR-10 classes

# ICSSR Deep Learning Track Submission

**Author:** Arya Singh Vishen

## Objective

Train a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories.

---

## Dataset

CIFAR-10 consists of:

* 60,000 color images (32×32 resolution)
* 10 object classes
* 50,000 training images
* 10,000 test images

---

## Train / Validation Split

An 80/20 split was applied to the official training dataset using PyTorch `random_split`:

* 40,000 images → Training set
* 10,000 images → Validation set

The official 10,000-image test set was kept unseen during training.

---

## Models Implemented

### Baseline CNN

* 2 Convolutional layers
* MaxPooling
* Fully Connected layers
* CrossEntropyLoss
* Adam optimizer

---

### Improved CNN

Enhancements applied:

* Data Augmentation

  * RandomHorizontalFlip
  * RandomCrop
* Normalization
* Additional Convolution Layer
* Batch Normalization
* Dropout Regularization

These changes were introduced to improve generalization and reduce overfitting.

---

## Metrics Reported

* Training Accuracy
* Validation Accuracy
* CrossEntropy Loss

---

## Best Results

### Baseline Model

* Training Accuracy: 95.25%
* Validation Accuracy: 70.14%

The large gap (~25%) between training and validation accuracy indicates overfitting.

---

### Improved Model

* Validation Accuracy: ~72–75%

The improved model demonstrated:

* More stable training curves
* Reduced overfitting
* Better generalization performance

---

## Analysis of Results

### Baseline Model

The baseline model achieved very high training accuracy but comparatively lower validation accuracy.
This suggests that the model memorized the training data rather than learning generalizable features.

---

### Improved Model

By introducing augmentation and regularization:

* The model became more robust to variations in image orientation and position.
* Batch Normalization stabilized training.
* Dropout reduced over-reliance on specific neurons.

As a result, validation performance improved, and the generalization gap decreased.

---

## Key Observations

* High training accuracy alone does not imply good performance.
* Validation accuracy is a better indicator of real-world performance.
* Data augmentation significantly improves robustness.
* Regularization techniques such as Dropout and BatchNorm help reduce overfitting.

Overall, the improved model demonstrates better balance between learning capacity and generalization.
