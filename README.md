**Andra Spoorthi (CS23B1007)**  
B.Tech – computer science   
 Indian Institute of Information Technology, Raichur 

# Project 1: Brain Tumor Segmentation using Thresholding

---

## Dataset

- **Dataset Type:** Brain MRI Tumor Dataset  
- **Annotation Format:** Binary Mask (.tif mask files)  
- **Data Used:** LGG MRI Segmentation Dataset  

### Kaggle Dataset Link  
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation  

---

## Methodology

### Ground Truth
Each MRI slice contains a corresponding binary mask file (`_mask.tif`).  
The mask represents tumor regions and is used for pixel-wise comparison with predicted segmentations.

---

### Segmentation Techniques

#### Otsu Thresholding (Global Method)
- Computes a single global threshold for the entire image.
- Separates tumor and background based on intensity histogram.
- Simple and computationally efficient.

#### Sauvola Thresholding (Adaptive Method)
- Computes local threshold values using neighborhood statistics.
- Designed for non-uniform illumination conditions.
- More flexible but sensitive to noise.

---

## Evaluation Metrics

Segmentation performance is evaluated per image and averaged across the dataset.

### Dice Score
Measures overlap between prediction and ground truth.

Dice = 2|A ∩ B| / (|A| + |B|)

### Jaccard Index (IoU)
Measures intersection over union.

Jaccard = |A ∩ B| / |A ∪ B|

---

# RESULTS

## OTSU THRESHOLDING (Global Method):

Average Dice:     0.1258  
Std Dice:         0.1011  
Average Jaccard:  0.0704  
Std Jaccard:      0.0611  

---

## SAUVOLA THRESHOLDING (Adaptive Method):

Average Dice:     0.0903  
Std Dice:         0.0694  
Average Jaccard:  0.0487  
Std Jaccard:      0.0399  

---

## COMPARISON

- Otsu performs better by +0.0355 (Dice Score).
- Otsu also achieves higher Jaccard values.
- Global thresholding worked better for this dataset compared to adaptive thresholding.

---

## Key Observations

- Tumor intensity sometimes overlaps with normal brain tissue, making thresholding difficult.
- Sauvola adaptive thresholding was sensitive to noise and fine brain structures.
- Otsu thresholding performed relatively better when tumor regions had distinguishable intensity contrast.
- Both methods produced low Dice scores, showing that classical thresholding is insufficient for complex medical image segmentation.

