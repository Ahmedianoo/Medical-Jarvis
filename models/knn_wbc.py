import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)



CSV_PATH = "wbc_table.csv"
RANDOM_STATE = 42
K_NEIGHBORS = 5
MIN_SAMPLES_PER_CLASS = 5



def parse_dict_safe(x):
    if not isinstance(x, str):
        return None
    try:
        x = x.replace("np.float64", "")
        x = x.replace("(", "")
        x = x.replace(")", "")
        return ast.literal_eval(x)
    except Exception:
        return None



df = pd.read_csv(CSV_PATH)
print("Original rows:", len(df))



df = df[
    (df["status"] != "image_not_found") &
    (
        (df["num_wbcs_detected"] == 1) |
        (df["has_multiple_gt"] == True)
    )
]

print("Rows after filtering:", len(df))



df["nucleus_features"] = df["nucleus_features"].apply(parse_dict_safe)
df["cytoplasm_features"] = df["cytoplasm_features"].apply(parse_dict_safe)

df = df.dropna(subset=["nucleus_features", "cytoplasm_features"])
print("Rows after parsing features:", len(df))



class_counts = df["ground_truth"].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df["ground_truth"].isin(valid_classes)]

print("\nClass distribution after filtering:")
print(df["ground_truth"].value_counts())



X = []
y = []
image_ids = []

for _, row in df.iterrows():
    n = row["nucleus_features"]
    c = row["cytoplasm_features"]

    X.append([
        n["area"],
        n["circularity"],
        n["solidity"],
        n["num_lobes"],
        c["mean_hue"],
        c["texture_variance"],
        c["cn_ratio"]
    ])
    y.append(row["ground_truth"])
    image_ids.append(row["image_number"])

X = np.array(X)
y = np.array(y)
image_ids = np.array(image_ids)



# split: 70 / 15 / 15 (With image ids)

X_train, X_temp, y_train, y_temp, img_train, img_temp = train_test_split(
    X, y, image_ids,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test, img_val, img_test = train_test_split(
    X_temp, y_temp, img_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# train
knn = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS,
    weights="distance"
)

knn.fit(X_train, y_train)


# save model and scalar

import joblib

joblib.dump(knn, "knn_wbc.pkl")
print("KNN model saved as knn_wbc.pkl")

joblib.dump(scaler, "scaler_wbc.pkl")
print("Scaler saved as scaler_wbc.pkl")

# evaluation
y_val_pred = knn.predict(X_val)
y_test_pred = knn.predict(X_test)

print("\nValidation Results:")
print(classification_report(y_val, y_val_pred))

print("\nTest Results:")
print(classification_report(y_test, y_test_pred))



cm = confusion_matrix(y_test, y_test_pred, labels=knn.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=knn.classes_
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("WBC Classification – Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix_knn_wbc.png", dpi=300)
plt.show()

print("\nConfusion matrix image saved as:")
print("confusion_matrix_knn_wbc.png")



print("\nTest set image numbers with predictions:")
for img_id, gt, pred in zip(img_test, y_test, y_test_pred):
    print(f"Image {img_id}: GT={gt}, Pred={pred}")
