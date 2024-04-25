import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Step 1: Load the data
X_train = np.load('exercice3/X_train.npy')
X_test = np.load('exercice3/X_test.npy')
y_train = np.load('exercice3/y_train.npy')
y_test = np.load('exercice3/y_test.npy')

# Step 2: Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Choose Classification Methods (Logistic Regression and SVM)
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC()
}

# Step 4: Optimization Procedures and Hyperparameter Tuning
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
params = {'Logistic Regression': param_grid_lr, 'SVM': param_grid_svm}

# Step 5: Cross-Validation
best_models = {}
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, params[name], cv=5)
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_

# Step 6: Train and Evaluate Models
for name, model in best_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')

# Step 7: Compare Results and Select Best Model
best_model_name = max(best_models, key=lambda k: accuracy_score(y_test, best_models[k].predict(X_test_scaled)))
best_model = best_models[best_model_name]
print(f'Best Model: {best_model_name}')

# Step 8: Discussion of Choices (Optional)
# Discuss the chosen hyperparameters, solvers, and cross-validation strategy

svm_model = SVC()

# Fit the model
svm_model.fit(X_train, y_train)

# Calculate predictions
y_pred = svm_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, svm_model.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()