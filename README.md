Title: Breast Cancer Classification with ML & Deep Learning

INITIAL COMMIT: This project builds a comprehensive machine learning pipeline to classify tumors as malignant (M) or benign (B) using the Breast Cancer Wisconsin (Diagnostic) dataset. The notebook/script includes data preprocessing, visualization, model training, evaluation, and advanced performance comparison.

SECOND COMMIT: Compared to the previous commit, this version introduces: Evaluation of six traditional ML models:  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - K-Nearest Neighbors (KNN)  
   - Naive Bayes  
   - Support Vector Machine (SVM)  
Cross-validation for KNN, NB, and SVM
Added confusion matrix heatmaps for each model, 
Visualized and compared: ROC curves and AUC scores, Precision-Recall curves and Average Precision scores

THIRD COMMIT: Compared to the previous commit, this version introduces: Saved figures for EDA and model evaluations (e.g. heatmaps, ROC/PR curves), Added deep learning models (1) Fully Connected Neural Network (MLP), (2)1D Convolutional Neural Network (CNN),  
Used early stopping for better generalization in neural models, Combined all results into a final performance summary table  

Dependencies

You can install all required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
                        OR
conda install pandas numpy matplotlib seaborn scikit-learn
conda install -c conda-forge tensorflow

