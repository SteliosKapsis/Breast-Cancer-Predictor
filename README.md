Breast Cancer Classification with ML & Deep Learning

This project builds a comprehensive machine learning and deep learning pipeline to classify tumors as malignant (M) or benign (B) using the Breast Cancer Wisconsin (Diagnostic) dataset. The notebook/script includes data preprocessing, visualization, model training, evaluation, and advanced performance comparison.

This version introduces:

Saved figures for **EDA and model evaluations** (e.g. heatmaps, ROC/PR curves)  
Evaluation of **six traditional ML models**:  
 - Logistic Regression  
 - Decision Tree  
 - Random Forest  
 - K-Nearest Neighbors (KNN)  
 - Naive Bayes  
 - Support Vector Machine (SVM)  
Cross-validation for KNN, NB, and SVM  
Added confusion matrix heatmaps for each model  
Visualized and compared:
 - ROC curves and AUC scores  
 - Precision-Recall curves and Average Precision scores  
Added **deep learning models:  
 - Fully Connected Neural Network (MLP)  
 - 1D Convolutional Neural Network (CNN)  
Used **early stopping** for better generalization in neural models  
Combined all results into a final **performance summary table**  

Dependencies

You can install all required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

or

conda install pandas numpy matplotlib seaborn scikit-learn
conda install -c conda-forge tensorflow

