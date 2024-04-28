import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt

# Step 1: Load the data
X_train = np.load('exercice4/X_train.npy')
X_test = np.load('exercice4/X_test.npy')
y_train = np.load('exercice4/y_train.npy').ravel()
y_test = np.load('exercice4/y_test.npy').ravel()

# Step 2: Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Choose Regression Methods (Ridge, Lasso, MLPRegressor, SVR, AdaBoostRegressor)
regressors = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'MLPRegressor': MLPRegressor(),
    'SVR': SVR(),
    'AdaBoostRegressor': AdaBoostRegressor()
}

# Step 4: Optimization Procedures and Hyperparameter Tuning
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
param_grid_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}
param_grid_mlp = {'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
                  'activation': ['relu', 'tanh'],
                  'alpha': [0.0001, 0.001, 0.01]}
param_grid_svr = {'kernel': ['linear', 'rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.2]}
param_grid_adaboost = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
params = {'Ridge': param_grid_ridge, 'Lasso': param_grid_lasso,
          'MLPRegressor': param_grid_mlp, 'SVR': param_grid_svr, 'AdaBoostRegressor': param_grid_adaboost}

# Step 5: Cross-Validation
best_regressors = {}
for name, reg in regressors.items():
    grid_search = GridSearchCV(reg, params[name], cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)
    best_regressors[name] = grid_search.best_estimator_

# Step 6: Train and Evaluate Models
for name, model in best_regressors.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} R2 Score: {r2:.4f}')

# Step 7: Compare Results and Select Best Model
best_model_name = max(best_regressors, key=lambda k: r2_score(y_test, best_regressors[k].predict(X_test_scaled)))
best_model = best_regressors[best_model_name]
print(f'Best Model: {best_model_name}')
