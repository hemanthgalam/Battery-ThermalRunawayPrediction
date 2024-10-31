import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.tree import export_graphviz
import graphviz

# Training function with Cross-Validation and GridSearchCV for hyperparameter tuning (now using regression)
def train_regression_model(data_path, model_save_path, scaler_save_path, hyperparam_save_path):
    df = pd.read_excel(data_path)

    # Here, instead of predicting stages, we predict the maximum temperature (or any other continuous value)
    df['Max_Temperature'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1)

    # Select features for the Random Forest model
    features = df[[
                    'Time',           # Original feature
                    'Voltage',        # Original feature
                    'T1',             # Original feature (Temperature sensor 1)
                    'T2',             # Original feature (Temperature sensor 2)
                    'T3',             # Original feature (Temperature sensor 3)
                    'T4',             # Original feature (Temperature sensor 4)
                    'T5',             # Original feature (Temperature sensor 5)
                    'T6',             # Original feature (Temperature sensor 6)
                    'Avg_Temp',       # Original feature (Average temperature)
                    'T1_diff',        # Gradient of T1
                    'T2_diff',        # Gradient of T2
                    'T3_diff',        # Gradient of T3
                    'T4_diff',        # Gradient of T4
                    'T5_diff',        # Gradient of T5
                    'T6_diff',        # Gradient of T6
                    'Voltage_diff'    # Gradient of Voltage
                ]]

    # The target variable for regression is now the maximum temperature
    X = features.values
    y = df['Max_Temperature'].values

    # Standardize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Set up hyperparameter tuning with GridSearchCV for regression
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [10, 20, 50],      # Maximum depth of the tree
        'min_samples_split': [2, 10, 20],      # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 5]     # Minimum samples required to be at a leaf node
    }
    # Initialize the RandomForestRegressor
    rf_regressor = RandomForestRegressor(random_state=42)

    # Use GridSearchCV for cross-validation and hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from the cross-validation
    best_model = grid_search.best_estimator_

    # Save the best hyperparameters to a file
    best_params = grid_search.best_params_
    with open(hyperparam_save_path, 'w') as f:
        json.dump(best_params, f)

    print(f"Best hyperparameters saved to {hyperparam_save_path}")
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation error: {grid_search.best_score_}")

    # Save the trained model and scaler
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"Model and scaler saved to {model_save_path} and {scaler_save_path}")

    # Evaluate the model on the test data
    y_pred = best_model.predict(X_test)

    # Compute regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test R^2: {r2}")
    print(f"Test MAE: {mae}")

    # Plot predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.5)
    plt.scatter(y_test, y_test, color='red', label='Actual', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'b--', label='Perfect Prediction')
    plt.xlabel('Actual Max Temperature')
    plt.ylabel('Predicted Max Temperature')
    plt.title('Actual vs Predicted Max Temperature (Test Set)')
    plt.grid(True)

    output_dir = 'overheat/training/results/rf_regression'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_rf_regressor.png'))
    plt.show()

# Testing function for regression
def test_regression_model(test_data_path, model_path, scaler_path):
    df = pd.read_excel(test_data_path)
    
    output_dir = 'overheat/testing/results/rf_regression'
    
    # The target variable for regression
    df['Max_Temperature'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1)

    # Load the trained Random Forest Regressor and scaler using pickle
    with open(model_path, 'rb') as model_file:
        rf_regressor = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Select features for the Random Forest model
    features = df[[
                    'Time',           # Original feature
                    'Voltage',        # Original feature
                    'T1',             # Original feature (Temperature sensor 1)
                    'T2',             # Original feature (Temperature sensor 2)
                    'T3',             # Original feature (Temperature sensor 3)
                    'T4',             # Original feature (Temperature sensor 4)
                    'T5',             # Original feature (Temperature sensor 5)
                    'T6',             # Original feature (Temperature sensor 6)
                    'Avg_Temp',       # Original feature (Average temperature)
                    'T1_diff',        # Gradient of T1
                    'T2_diff',        # Gradient of T2
                    'T3_diff',        # Gradient of T3
                    'T4_diff',        # Gradient of T4
                    'T5_diff',        # Gradient of T5
                    'T6_diff',        # Gradient of T6
                    'Voltage_diff'    # Gradient of Voltage
                ]]

    # Standardize features using the loaded scaler
    X_scaled = scaler.transform(features.values)

    # Predict on the test dataset
    y_pred = rf_regressor.predict(X_scaled)

    # Evaluate the model
    y_test = df['Max_Temperature'].values
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test R^2: {r2}")
    print(f"Test MAE: {mae}")
    
    test_metrics = {
        "Test MSE": mse,
        "Test R^2": r2,
        "Test MAE": mae
    }

    # Save test metrics to a JSON file
    test_metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
        

    feature_names = ['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Avg_Temp', 
                    'T1_diff', 'T2_diff', 'T3_diff', 'T4_diff', 'T5_diff', 'T6_diff', 'Voltage_diff']
    plot_feature_importance(rf_regressor, feature_names, output_dir)
    
    save_tree_plot(rf_regressor, feature_names, output_dir)

    # Plot predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.5)
    plt.scatter(y_test, y_test, color='red', label='Actual', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'b--', label='Perfect Prediction')
    plt.xlabel('Actual Max Temperature')
    plt.ylabel('Predicted Max Temperature')
    plt.title('Actual vs Predicted Max Temperature (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_rf_testing.png'))
    plt.show()
    
def save_tree_plot(rf_model, feature_names, output_dir, tree_index=0, max_depth=3):
    # Extract the specified tree from the RandomForestRegressor
    estimator = rf_model.estimators_[tree_index]
    
    # Export the tree to a dot file
    dot_file = os.path.join(output_dir, f'tree_{tree_index}.dot')
    export_graphviz(estimator, out_file=dot_file, 
                    feature_names=feature_names, 
                    filled=True, rounded=True, 
                    special_characters=True, 
                    max_depth=max_depth)

    # Convert the dot file to a PNG image
    with open(dot_file) as f:
        dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    graph.render(os.path.join(output_dir, f'tree_{tree_index}'), format="png", cleanup=True)

    print(f"Tree {tree_index} saved as PNG in {output_dir}")


def plot_feature_importance(model, feature_names, output_dir):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_RF.png'))
    plt.show()



# Main script execution
if __name__ == "__main__":
    # File paths for training and testing
    training_data_path = 'overheat/training/regression_data.xlsx'
    test_data_path = 'overheat/testing/regression_data.xlsx'
    
    model_save_path = 'overheat/trained_models/rf_regression/random_forest_regressor.pkl'
    scaler_save_path = 'overheat/trained_models/rf_regression/scaler.pkl'
    hyperparam_save_path = 'overheat/trained_models/rf_regression/best_hyperparameters_rf_regressor.json'

    # Train the regression model with hyperparameter tuning
    train_regression_model(training_data_path, model_save_path, scaler_save_path, hyperparam_save_path)

    # Test the regression model
    test_regression_model(test_data_path, model_save_path, scaler_save_path)
