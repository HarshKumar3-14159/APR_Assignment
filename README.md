# SVM Classifier for Handwritten Digit Recognition 
This project demonstrates the implementation of a Support Vector Machine (SVM) classifier to recognize handwritten digits. The model is built and evaluated using the classic Digits dataset, which is available in the scikit-learn library. The entire workflow is contained within a single Google Colab notebook.

### Overview
The primary goal of this project is to showcase a complete machine learning pipeline:

Loading and exploring a dataset.

Preprocessing the data for the model.

Training a powerful classification algorithm (SVM).

Evaluating the model's performance using multiple metrics.

Visualizing the model's training progress and results.

### Dataset
Name: Handwritten Digits Dataset

Source: sklearn.datasets.load_digits

Description: The dataset consists of 1,797 grayscale images of handwritten digits from 0 to 9.

Image Size: Each image is 8x8 pixels.

Features: Each image is flattened into a feature vector of 64 pixel intensity values.

Target: A label from 0 to 9 corresponding to the digit in the image.

### Dependencies
This project is designed to run in a standard Google Colab environment. The core libraries used are:

scikit-learn

numpy

matplotlib

seaborn

These libraries are pre-installed in Google Colab, so no additional installation is required.

### How to Run
Open the .ipynb notebook file in Google Colab.

Navigate to the menu and select Runtime > Run all.

The notebook will execute all cells sequentially, from data loading to final evaluation, and display all outputs and visualizations.

🛠️ Project Workflow
The notebook is structured into several key sections that follow a standard machine learning workflow.

1. Data Loading and Exploration
The Digits dataset is loaded from scikit-learn. Initial visualizations are created to display sample images and their corresponding labels, providing an intuitive understanding of the data.

2. Preprocessing and Splitting
Feature Scaling: The 64 pixel features are scaled using StandardScaler. This is a crucial step for SVMs, as they are sensitive to the scale of input features. Scaling ensures that all features contribute equally to the distance calculations.

Data Splitting: The dataset is split into a training set (80%) and a testing set (20%) to ensure the model is evaluated on unseen data.

3. Model Training
A Support Vector Classifier (svm.SVC) is used for this multi-class classification task.

Kernel: The rbf (Radial Basis Function) kernel is a common choice, but a linear kernel or others could also be used.

Parameters: The gamma parameter was set to 0.001, and probability=True was enabled to allow for the calculation of prediction probabilities needed for the ROC curve.

4. Model Evaluation
The trained model's performance is assessed using several metrics:

Accuracy Score: Provides a single percentage representing the overall correctness of the model's predictions.

Confusion Matrix: A heatmap is generated to visualize the model's performance on a per-class basis. It clearly shows which digits, if any, were commonly confused with others.

ROC Curve & AUC: Since this is a multi-class problem, One-vs-Rest (OvR) ROC curves are plotted for each of the 10 classes. This visualizes the model's ability to distinguish each digit from all the others.

5. Visualizing Training Progress
A Learning Curve is generated to visualize how the model's performance on both the training and validation sets changes as the number of training samples increases. This is a powerful tool for diagnosing high variance (overfitting) or high bias (underfitting).

### Results
The SVM model performs exceptionally well on this task, achieving an accuracy of approximately 99% on the test set.

The confusion matrix confirms this high performance, showing that nearly all test images are correctly classified, with very few off-diagonal entries.

The learning curve shows that the training and cross-validation scores converge to a high value, indicating that the model generalizes well and is not overfitting.

The ROC curves for all classes are close to the top-left corner, with Area Under the Curve (AUC) values near 1.0, demonstrating the model's excellent discriminative ability for all digits.

This project successfully demonstrates the effectiveness of SVMs for image classification tasks.
