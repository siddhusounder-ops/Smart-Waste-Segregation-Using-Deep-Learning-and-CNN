# Smart Waste Segregation Using Deep Learning and CNN
Waste Material Segregation using CNNs

A convolutional neural network for classifying waste materials into 7 categories to support automated recycling and waste sorting systems.

Overview

This project builds an image classification pipeline that categorizes waste into Cardboard, Food Waste, Glass, Metal, Other, Paper, and Plastic, aiming to improve recycling efficiency and reduce landfill contamination through automated sorting.

Dataset
7,625 images across 7 classes, loaded from a directory structure organized by class label
Notable class imbalance: Plastic (2,295 images, majority) vs. Cardboard (540 images, minority)
Images resized to 128×128 and normalized to [0, 1]
80/20 stratified train/validation split
Approach
Preprocessing: Custom image loading/resizing pipeline with error handling; label encoding via LabelEncoder
Model: Sequential CNN with 3 convolutional blocks (Conv2D → BatchNorm → MaxPooling → Dropout), followed by fully connected layers and a softmax output
Training: 20 epochs, batch size 32, Adam optimizer, sparse categorical crossentropy
Evaluation: Classification report + confusion matrix per class
Augmentation experiment: Tested RandomFlip, RandomRotation, RandomZoom, and RandomBrightness to address overfitting
Results
The base CNN hit 93.7% training accuracy but only 60.9% validation accuracy (loss: 1.46), indicating overfitting. Recall varied by class — strong for Plastic (0.81), weak for Metal and Other (0.41, 0.38).
Adding data augmentation (flips, rotation, zoom, brightness) to fix the overfitting backfired: accuracy dropped to ~30% on both train and validation, near-random for 7 classes. Likely causes: overly aggressive augmentation parameters or a weight re-initialization issue during retraining.
Key Takeaways
Class imbalance and inter-class visual similarity (e.g., Metal vs. Other) are the main drivers of weak spots in the classifier
Data augmentation is not a free win — parameter tuning and careful retraining setup matter more than simply adding transforms
Next steps: class-weighted loss, transfer learning from a pretrained backbone (e.g., MobileNetV2/ResNet), and more conservative augmentation
Tech Stack

Python · TensorFlow/Keras · scikit-learn · NumPy/Pandas · Matplotlib/Seaborn
