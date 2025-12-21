import joblib
import numpy as np
import cv2
import sys
import os
from typing import List, Dict, Optional
from pprint import pprint

from wbc_features import WBCClassifier
from preprocess import preprocess_img
from segment import wbc



class KNNWBCClassifier(WBCClassifier):
    """
    WBC classifier using a trained KNN model instead of thresholds
    """

    def __init__(self, knn_path="knn_wbc.pkl", scaler_path="scaler_wbc.pkl"):
        super().__init__()

        self.knn = joblib.load(knn_path)
        self.scaler = joblib.load(scaler_path)

        self.feature_order = [
            "area",
            "circularity",
            "solidity",
            "num_lobes",
            "mean_hue",
            "texture_variance",
            "cn_ratio"
        ]

    def classify_wbc(self, nf, cf) -> str:
        """
        Classify a single WBC using KNN
        """
        if nf is None or cf is None:
            return "BASOPHIL"  # safe fallback

        feature_vector = np.array([[
            nf["area"],
            nf["circularity"],
            nf["solidity"],
            nf["num_lobes"],
            cf["mean_hue"],
            cf["texture_variance"],
            cf["cn_ratio"]
        ]])
        # feature_vector = np.round(feature_vector, 5)
        feature_vector = self.scaler.transform(feature_vector)
        prediction = self.knn.predict(feature_vector)[0]

        return prediction

# knn_classifier = KNNWBCClassifier()
# img = cv2.imread("../data/input/JPEGImages/BloodImage_00313.jpg")
# pre = preprocess_img(img)
# wbc_mask = wbc(pre)
# res = knn_classifier.classify_all_wbcs(pre, wbc_mask)
# pprint(res)