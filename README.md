# Deep Learning Model for Chest X-Ray Classification (COVID-19, Pneumonia, Normal)
This repository contains Python code for training a deep learning model to classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal. The model leverages a pre-trained VGG16 convolutional neural network (CNN) for feature extraction and a custom head for classification.

Key functionalities:

* Loads chest X-ray images from designated folders.
* Splits data into training, validation, and test sets with stratification to ensure balanced class representation.
* Applies data augmentation techniques (rotation, zoom, flip) to the training data for improved generalization.

Model architecture:

* Utilizes a pre-trained VGG16 model with weights frozen for feature extraction.
* Adds a custom classification head with dense layers and dropout for regularization.
* Employs categorical cross-entropy loss and Adam optimizer for training.

Training, evaluation & prediction:
* Implements early stopping to monitor validation accuracy and prevent overfitting.
* Generates training and validation data using data generators for efficient memory usage.
* Plots training and validation loss/accuracy curves to visualize model performance.
* Predicts the class probability for the test image.

## Further considerations:
This is a basic implementation and can be extended with hyperparameter tuning and different network architectures.
The model performance can be improved with a larger and more diverse dataset.
Explore incorporating techniques like transfer learning with fine-tuning all model layers for potentially better results.

Note: This model is for educational purposes only and should not be used for medical diagnosis.
