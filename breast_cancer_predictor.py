import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    f1_score
)

warnings.filterwarnings('ignore')


df = pd.read_csv('data.csv')
df.head()
df.tail()
df.shape
df.describe().T
df.diagnosis.unique()

print(df['diagnosis'].value_counts())
sns.countplot(df['diagnosis'], palette='husl')
plt.savefig("diagnosis_count.png")
plt.close()

df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
df.head()
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()
df.isnull().sum()
df.corr()

plt.hist(df['diagnosis'], color='g')
plt.title('Plot_Diagnosis (M=1 , B=0)')
plt.show()
plt.savefig("Plot_Diagnosis.png")
plt.close()

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
plt.savefig("heat_map.png")
plt.close()

# generate a scatter plot matrix with the "mean" columns
cols = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']

sns.pairplot(data=df[cols], hue='diagnosis', palette='rocket')
plt.savefig(" scatter plot matrix with the ""mean"" columns.png")
plt.close()

# Generate and visualize the correlation matrix
corr = df.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(20, 20))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.savefig(" correlation matrix.png")
plt.close()

plt.tight_layout()

# first, drop all "worst" columns
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
df = df.drop(cols, axis=1)

# then, drop all columns related to the "perimeter" and "area" attributes
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)

# lastly, drop all columns related to the "concavity" and "concave points" attributes
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)

# verify remaining columns
df.columns

# Draw the heatmap again, with the new correlation matrix
corr = df.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.savefig("correlation matrix after data cleaning.png")
plt.close()

plt.tight_layout()

X=df.drop(['diagnosis'],axis=1)
y = df['diagnosis']

from sklearn.model_selection import train_test_split

# Split data
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Logistic Regression ----------------
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

cm_lr = confusion_matrix(y_test, pred_lr)
sns.heatmap(cm_lr, annot=True)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig('logistic_regression_cm.png')
plt.close()

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# ---------------- Decision Tree ----------------
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
pred_dtc = dtc.predict(X_test)

cm_dtc = confusion_matrix(y_test, pred_dtc)
sns.heatmap(cm_dtc, annot=True)
plt.title("Confusion Matrix - Decision Tree")
plt.savefig('decision_tree_cm.png')
plt.close()

print("\nDecision Tree Accuracy:", accuracy_score(y_test, pred_dtc))
print(classification_report(y_test, pred_dtc))

# ---------------- Random Forest ----------------
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

cm_rfc = confusion_matrix(y_test, pred_rfc)
sns.heatmap(cm_rfc, annot=True)
plt.title("Confusion Matrix - Random Forest")
plt.savefig('random_forest_cm.png')
plt.close()

print("\nRandom Forest Accuracy:", accuracy_score(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))

# ---------------- KNN, Naive Bayes, SVM (Cross-Validation) ----------------
models = [
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

print("\nCross-Validation Results:")
for name, model in models:
    kfold = KFold(n_splits=10)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

cm_knn = confusion_matrix(y_test, KNeighborsClassifier().fit(X_train, y_train).predict(X_test))
sns.heatmap(cm_knn, annot=True)
plt.title("Confusion Matrix - KNN")
plt.savefig('knn_cm.png')
plt.close()


cm_nb = confusion_matrix(y_test, GaussianNB().fit(X_train, y_train).predict(X_test))
sns.heatmap(cm_nb, annot=True)
plt.title("Confusion Matrix - Naive Bayes")
plt.savefig('naive_bayes_cm.png')
plt.close()


cm_svm = confusion_matrix(y_test, pred_svm)
sns.heatmap(cm_svm, annot=True)
plt.title("Confusion Matrix - SVM")
plt.savefig('svm_cm.png')
plt.close()


# ---------------- Final SVM Evaluation ----------------
final_svm = SVC()
final_svm.fit(X_train, y_train)
pred_svm = final_svm.predict(X_test)

print("\nFinal SVM Accuracy:", accuracy_score(y_test, pred_svm))
print(classification_report(y_test, pred_svm))
print("Confusion Matrix - SVM:\n", confusion_matrix(y_test, pred_svm))

from sklearn.metrics import precision_recall_curve, average_precision_score

# Create a new figure for PR curves
plt.figure(figsize=(10, 8))

# Dictionary to store average precision scores
pr_scores = {}

# Helper to plot PR curve
def plot_pr_curve(model, X_test, y_test, label):
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    pr_scores[label] = avg_precision
    plt.plot(recall, precision, lw=2, label=f'{label} (AP = {avg_precision:.2f})')


# Re-fit classifiers if needed (ensure probability support)
models_pr = {
    "Logistic Regression": LogisticRegression().fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier().fit(X_train, y_train),
    "Random Forest": RandomForestClassifier().fit(X_train, y_train),
    "KNN": KNeighborsClassifier().fit(X_train, y_train),
    "Naive Bayes": GaussianNB().fit(X_train, y_train),
    "SVM": SVC(probability=True).fit(X_train, y_train)
}


for name, model in models_pr.items():
    plot_pr_curve(model, X_test, y_test, name)

# Plot formatting
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for All Models")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("Precision-Recall_All.png")
plt.close()


from sklearn.metrics import roc_curve, auc

# Create a figure
plt.figure(figsize=(10, 8))

# Dictionary to store AUC scores
auc_scores = {}

# Helper to plot ROC curve
def plot_roc_curve(model, X_test, y_test, label):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # For SVM (no predict_proba by default), use decision_function
        y_score = model.decision_function(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    auc_scores[label] = roc_auc
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')


# Re-fit classifiers to ensure we have access to their probability outputs
models_roc = {
    "Logistic Regression": LogisticRegression().fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier().fit(X_train, y_train),
    "Random Forest": RandomForestClassifier().fit(X_train, y_train),
    "KNN": KNeighborsClassifier().fit(X_train, y_train),
    "Naive Bayes": GaussianNB().fit(X_train, y_train),
    "SVM": SVC(probability=True).fit(X_train, y_train)  # important to enable probability estimates
}


for name, model in models_roc.items():
    plot_roc_curve(model, X_test, y_test, name)

# Plot formatting
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("ROC_curves_All.png")
plt.close()


from sklearn.metrics import f1_score

# Reuse models already trained or re-fit them here
models_metrics = {
    "Logistic Regression": LogisticRegression().fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier().fit(X_train, y_train),
    "Random Forest": RandomForestClassifier().fit(X_train, y_train),
    "KNN": KNeighborsClassifier().fit(X_train, y_train),
    "Naive Bayes": GaussianNB().fit(X_train, y_train),
    "SVM": SVC(probability=True).fit(X_train, y_train)
}

# Create empty list to hold results
metrics_summary = []

# Collect metrics
for name, model in models_metrics.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
        
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    auc_score = auc(*roc_curve(y_test, y_score)[:2])
    ap_score = average_precision_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred)
    
    metrics_summary.append([name, acc, auc_score, ap_score, f1])

# Convert to DataFrame
ml_summary_df = pd.DataFrame(metrics_summary, columns=['Model', 'Accuracy', 'AUC', 'Average Precision', 'F1 Score'])

# Round values for cleaner output
ml_summary_df[['Accuracy', 'AUC', 'Average Precision', 'F1 Score']] = ml_summary_df[['Accuracy', 'AUC', 'Average Precision', 'F1 Score']].round(4)


# Display summary table
print("Model Performance Summary:")
print(ml_summary_df)

# #1 SVM and # 2 KNN and #3 Logistic Regression


# In[96]:


############################################
########### DEEP LEARNING MODELS ###########
############################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

# ---------------- Deep Learning Model (MLP) ----------------
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nNeural Network Accuracy: {accuracy:.4f}")

y_pred_nn = model.predict(X_test).ravel()
y_pred_label = (y_pred_nn > 0.5).astype(int)

acc_nn = accuracy_score(y_test, y_pred_label)
auc_nn = roc_auc_score(y_test, y_pred_nn)
ap_nn = average_precision_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_label)

print(f"Neural Network AUC: {auc_nn:.4f}")
print(f"Neural Network Average Precision: {ap_nn:.4f}")
print(f"Neural Network F1 Score: {f1_nn:.4f}")


metrics_summary.append(['Neural Network', acc_nn, auc_nn, ap_nn, f1_nn])


plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Neural Network Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("NN Training Loss.png")
plt.close()


# ---------------- Deep Learning Model (CNN) ----------------
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Reshape for Conv1D: (samples, time_steps, features)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# ---------------- Convolutional Neural Network ----------------
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(16, activation='relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop_cnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_cnn = cnn_model.fit(X_train_cnn, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop_cnn])

# Evaluate
loss_cnn, acc_cnn = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nCNN Accuracy: {acc_cnn:.4f}")


y_pred_cnn_prob = cnn_model.predict(X_test_cnn).ravel()
y_pred_cnn_label = (y_pred_cnn_prob > 0.5).astype(int)

auc_cnn = roc_auc_score(y_test, y_pred_cnn_prob)
ap_cnn = average_precision_score(y_test, y_pred_cnn_prob)
f1_cnn = f1_score(y_test, y_pred_cnn_label)

print(f"CNN AUC: {auc_cnn:.4f}")
print(f"CNN Average Precision: {ap_cnn:.4f}")
print(f"CNN F1 Score: {f1_cnn:.4f}")


metrics_summary.append(['CNN', acc_cnn, auc_cnn, ap_cnn, f1_cnn])


plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Val Loss')
plt.title('CNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("CNN_Training_Loss.png")
plt.close()


# Create a summary DataFrame for deep learning models
dl_metrics = [
    ['Neural Network (MLP)', acc_nn, auc_nn, ap_nn, f1_nn],
    ['Convolutional Neural Network (CNN)', acc_cnn, auc_cnn, ap_cnn, f1_cnn]
]

dl_summary_df = pd.DataFrame(dl_metrics, columns=['Model', 'Accuracy', 'AUC', 'Average Precision', 'F1 Score'])

# Round for cleaner display
dl_summary_df[['Accuracy', 'AUC', 'Average Precision', 'F1 Score']] = dl_summary_df[['Accuracy', 'AUC', 'Average Precision', 'F1 Score']].round(4)

# Display the table
print("\nDeep Learning Model Performance Summary:")
print(dl_summary_df)


# Combine ML and DL summaries
all_models_df = pd.concat([ml_summary_df, dl_summary_df], ignore_index=True)

# Sort by F1 Score (or any metric you prefer)
all_models_df = all_models_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

# Display the table
print("\nCombined ML + DL Model Performance Summary:")
print(all_models_df)

