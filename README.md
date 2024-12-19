# Project Title: Knee OA Classification Using Bone Distance

## Overview
This project automates the classification of Knee Osteoarthritis (OA) severity by measuring femur-tibia distances from MRI slices and training machine learning models. The approach leverages image preprocessing, centroid distance measurement, and feature engineering to create a reliable framework for OA diagnosis.

## Features
- **Preprocessing**: Grayscale conversion, noise reduction, and edge detection for contour identification.
- **Distance Calculation**: Centroid-based Euclidean distance measurement between femur and tibia contours.
- **Feature Engineering**: Dimensionality reduction using PCA and class balancing with SMOTE.
- **Model Training**: Implementation of Random Forest, Logistic Regression, and Multi-Layer Perceptron classifiers.
- **Evaluation**: 10-fold cross-validation and ROC analysis to assess model performance.

## Requirements
To run this project, you will need the following Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

- WEKA (optional for machine learning classification)

## How to Run
1. Ensure all dependencies are installed.
2. Download the bone mask database (197 cases) and MRI images.
3. Execute the steps below to complete the project.

### Key Steps
1. **Image Preprocessing**:
    - Convert images to grayscale to reduce complexity.
    - Apply bilateral filtering to preserve edges while reducing noise.
    - Detect edges using the Canny edge detection method.
    - Extract contours of femur and tibia bones using OpenCV.
    - Calculate centroids for each contour to determine bone positions.
2. **Distance Measurement**:
    - Compute Euclidean distances between femur and tibia centroids for each valid slice.
    - Exclude invalid slices without clear bone structures.
3. **Model Training**:
    - Train classifiers including Random Forest, Logistic Regression, and MLP.
    - Perform 10-fold cross-validation for evaluation.
4. **Feature Engineering**:
    - Apply Principal Component Analysis (PCA) to reduce dimensions while retaining variance.
    - Use SMOTE to balance the dataset and address class imbalance.
5. **Optimization**:
    - Experiment with alternative distance measurement strategies and feature representations.
    - Evaluate performance using ROC-AUC and accuracy metrics.
6. **Final Analysis**:
    - Compare models and feature representations.
    - Document findings and prepare a presentation.

## Results
- **Cross-Validation Accuracy**: Achieved an average of 71% across 10 folds.
- **AUC Score**: The ROC analysis showed a moderate AUC score of 0.63.

## Example Output
- **ROC Curve Plot**: A visual representation of model performance.
- **AUC Score**: Quantitative measure of model accuracy.

## Challenges and Future Work
### Challenges
- **Dataset Size**: Limited data affected model generalization.
- **Noise and Variability**: Variability in MRI quality impacted contour detection.
- **Class Imbalance**: Addressed using SMOTE to balance OA and Non-OA samples.


## File Structure
- `Final_Notebook.ipynb`: Jupyter Notebook containing implementation steps.
- `README.md`: Instructions and project documentation (this file).
- Bone Mask Database: Collection of segmented bone mask images (to be downloaded).


