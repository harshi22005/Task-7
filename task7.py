import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# ---------------------------
# LOAD DATA FROM ZIP
# ---------------------------
zip_path = r"C:\\Users\\G HARSHITHA\\Downloads\\archive (8).zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    file = [f for f in z.namelist() if f.endswith('.csv')][0]  # pick CSV inside zip
    df = pd.read_csv(z.open(file))

print("\nDataset Loaded Successfully!\n")
print(df.head(), "\n")

# ---------------------------
# PREPROCESSING
# ---------------------------
df = df.drop(columns=["id"])          # remove id column
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # encode labels

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# MODEL 1: LINEAR SVM
# ---------------------------
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, linear_pred))
print("\nLinear SVM Confusion Matrix:\n", confusion_matrix(y_test, linear_pred))

# ---------------------------
# MODEL 2: RBF SVM
# ---------------------------
rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)

print("\nRBF SVM Accuracy:", accuracy_score(y_test, rbf_pred))
print("\nRBF SVM Confusion Matrix:\n", confusion_matrix(y_test, rbf_pred))

# ---------------------------
# HYPERPARAMETER TUNING
# ---------------------------
params = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 0.01]
}

grid = GridSearchCV(SVC(kernel="rbf"), params, cv=5)
grid.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# ---------------------------
# CROSS-VALIDATION
# ---------------------------
scores = cross_val_score(SVC(kernel="rbf", C=grid.best_params_["C"],
                             gamma=grid.best_params_["gamma"]),
                             X_scaled, y, cv=5)

print("\nCross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())

# ---------------------------
# DIMENSION REDUCTION (PCA) FOR PLOTTING
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fit SVM again on PCA-reduced data
model_plot = SVC(kernel="rbf", C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
model_plot.fit(X_pca, y)

# ---------------------------
# DECISION BOUNDARY PLOT
# ---------------------------
plt.figure(figsize=(8,6))

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = model_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolors='k')

plt.title("SVM Decision Boundary (PCA 2D Projection)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()
