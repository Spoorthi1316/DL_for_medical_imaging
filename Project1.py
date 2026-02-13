import os
import numpy as np
from skimage import io, filters
from sklearn.metrics import jaccard_score

dataset_path = "kaggle_3m"  
dice_otsu_scores = []
dice_sauvola_scores = []

jaccard_otsu_scores = []
jaccard_sauvola_scores = []

sample_image = None
sample_mask = None
sample_otsu = None
sample_sauvola = None

#Accessing dataset
for patient in os.listdir(dataset_path):
    patient_path = os.path.join(dataset_path, patient)
    if not os.path.isdir(patient_path):
        continue
    print("Processing patient:", patient)
    for file in os.listdir(patient_path):
        if "_mask" in file:
            continue
        image_path = os.path.join(patient_path, file)
        if image_path.endswith(".TIF"):
            mask_path = image_path.replace(".TIF", "_mask.TIF")
        else:
            mask_path = image_path.replace(".tif", "_mask.tif")
        if not os.path.exists(mask_path):
            continue
        image = io.imread(image_path)
        mask = io.imread(mask_path)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        image = image / 255.0
        # Converting mask to binary
        mask = (mask > 0).astype(np.uint8)
        # Skipping slices with NO tumor
        if np.sum(mask) == 0:
            continue

        # OTSU 
        otsu_thresh = filters.threshold_otsu(image)
        otsu_seg = (image > otsu_thresh).astype(np.uint8)
        # SAUVOLA 
        window_size = 25
        sauvola_thresh = filters.threshold_sauvola(image, window_size=window_size)
        sauvola_seg = (image > sauvola_thresh).astype(np.uint8)
    
        mask_flat = mask.flatten()
        otsu_flat = otsu_seg.flatten()
        sauvola_flat = sauvola_seg.flatten()
        # DICE 
        intersection_otsu = np.sum(mask_flat * otsu_flat)
        dice_otsu = (2. * intersection_otsu) / (np.sum(mask_flat) + np.sum(otsu_flat))
        dice_otsu_scores.append(dice_otsu)

        intersection_sauvola = np.sum(mask_flat * sauvola_flat)
        dice_sauvola = (2. * intersection_sauvola) / (np.sum(mask_flat) + np.sum(sauvola_flat))
        dice_sauvola_scores.append(dice_sauvola)

        # JACCARD
        jaccard_otsu = jaccard_score(mask_flat, otsu_flat)
        jaccard_sauvola = jaccard_score(mask_flat, sauvola_flat)

        jaccard_otsu_scores.append(jaccard_otsu)
        jaccard_sauvola_scores.append(jaccard_sauvola)
        
#Results
print("RESULTS")
print("\n OTSU THRESHOLDING (Global Method):")
print(f"  Average Dice:    {np.mean(dice_otsu_scores):.4f}")
print(f"  Std Dice:        {np.std(dice_otsu_scores):.4f}")
print(f"  Average Jaccard: {np.mean(jaccard_otsu_scores):.4f}")
print(f"  Std Jaccard:     {np.std(jaccard_otsu_scores):.4f}")

print("\n SAUVOLA THRESHOLDING (Adaptive Method):")
print(f"  Average Dice:    {np.mean(dice_sauvola_scores):.4f}")
print(f"  Std Dice:        {np.std(dice_sauvola_scores):.4f}")
print(f"  Average Jaccard: {np.mean(jaccard_sauvola_scores):.4f}")
print(f"  Std Jaccard:     {np.std(jaccard_sauvola_scores):.4f}")

print("\n COMPARISON:")
dice_diff = np.mean(dice_sauvola_scores) - np.mean(dice_otsu_scores)
jaccard_diff = np.mean(jaccard_sauvola_scores) - np.mean(jaccard_otsu_scores)

if dice_diff > 0:
    print(f"  Sauvola performs better by +{dice_diff:.4f} (Dice)")
else:
    print(f"  Otsu performs better by +{-dice_diff:.4f} (Dice)")
