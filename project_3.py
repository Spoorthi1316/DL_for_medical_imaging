import os
import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola

image_folder = "Training/Images"
mask_folder = "Training/Masks"

def sensitivity(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    TP = np.sum((pred == 1) & (gt == 1))
    FN = np.sum((pred == 0) & (gt == 1))
    if (TP + FN) == 0:
        return 0
    return TP / (TP + FN)

windows = [15, 25]
niblack_k_values = [-0.2, 0.0, 0.2]
savoula_k_values = [0.2, 0.3]
best_niblack_score = 0
best_niblack_params = None
best_sauvola_score = 0
best_sauvola_params = None

for w in windows:
    for k in niblack_k_values:
        sens_list = []
        for image_name in os.listdir(image_folder):
            if "HRF" not in image_name:
                continue
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = image[:, :, 1]
            name = os.path.splitext(image_name)[0]
            mask_path = os.path.join(mask_folder, name + ".tif")
            if not os.path.exists(mask_path):
                continue
            mask = cv2.imread(mask_path, 0)
            mask = mask > 0
            thresh = threshold_niblack(image, window_size=w, k=k)
            binary = image < thresh
            sens = sensitivity(binary, mask)
            sens_list.append(sens)
        avg_sens = np.mean(sens_list)
        print(f"Niblack -> window={w}, k={k}, sensitivity={avg_sens:.4f}")
        if avg_sens > best_niblack_score:
            best_niblack_score = avg_sens
            best_niblack_params = (w, k)

for w in windows:
    for k in savoula_k_values:
        sens_list = []
        for image_name in os.listdir(image_folder):
            if "HRF" not in image_name:
                continue
            image = cv2.imread(os.path.join(image_folder, image_name))
            image = image[:, :, 1]
            name = os.path.splitext(image_name)[0]
            mask = cv2.imread(os.path.join(mask_folder, name + ".tif"), 0)
            mask = mask > 0
            thresh = threshold_sauvola(image, window_size=w, k=k)
            binary = image < thresh
            sens = sensitivity(binary, mask)
            sens_list.append(sens)
        avg = np.mean(sens_list)
        print("Window:", w, "k:", k, "Sensitivity:", avg)
        if avg > best_sauvola_score:
            best_sauvola_score = avg
            best_sauvola_params = (w, k)

print("Niblack: ")
print("Best Niblack Parameters:", best_niblack_params)
print("Best Niblack Sensitivity:", best_niblack_score)
print("\nSauvola: ")
print("Best Sauvola Window:", best_sauvola_params)
print("Best Sauvola Sensitivity:", best_sauvola_score)
