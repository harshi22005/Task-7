Task 7 – Support Vector Machines (SVM) Classification

This repository contains the implementation of Support Vector Machines (SVM) for binary classification using the Breast Cancer Wisconsin Dataset. The task demonstrates loading and preprocessing data, training SVM models, visualizing decision boundaries, and evaluating performance.

📘 Project Description

Support Vector Machines (SVM) are powerful supervised learning models used for classification tasks.
In this project, SVM is applied to classify tumors as Benign (B) or Malignant (M).

The workflow includes dataset handling, feature scaling, model training using both Linear and RBF kernels, and visualization of the separation boundary using two primary features.

📂 Dataset

Breast Cancer Wisconsin Dataset
File: data.csv (inside the provided ZIP folder)

The dataset includes:

30 numerical features describing tumor characteristics

A target label:

M – Malignant

B – Benign

🧠 Features of the Implementation

Loads dataset from ZIP-extracted folder

Encodes categorical labels

Scales the features using StandardScaler

Trains SVM using:

Linear Kernel

RBF Kernel

Evaluates the model using:

Accuracy

Confusion Matrix

Visualizes the decision boundary using two features:

radius_mean

texture_mean

🚀 How to Run the Program

Place:

task7.py

Dataset folder: archive (8) (containing data.csv)
in the same directory.

Run the script using:

python task7.py


The output will display:

Dataset loading confirmation

Accuracy score

Confusion matrix

Decision boundary graph

📊 Output Expected

Printed accuracy score

Confusion matrix table

Scatter plot showing decision boundary
(Malignant vs. Benign regions using two features)

🛠️ Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

👩‍🎓 Author
G Harshitha
AIML Student
