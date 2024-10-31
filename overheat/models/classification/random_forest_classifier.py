import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.tree import export_graphviz, export_text
import graphviz
from sklearn.tree import plot_tree

# Training function with Cross-Validation and GridSearchCV for hyperparameter tuning
def train_random_forest_classifier(data_path, model_save_path, scaler_save_path, encoder_save_path, hyperparam_save_path):
    df = pd.read_excel(data_path)

    # Compute gradients for temperature sensors and voltage
    for i in range(1, 7):
        temp_col = f'T{i}'
        df[f'{temp_col}_diff'] = df[temp_col].diff()

    df['Voltage_diff'] = df['Voltage'].diff()
    df.fillna(0, inplace=True)

    # Use the `Class` column directly as the target for classification
    target = 'Class'

    # Encode target labels if they are categorical
    label_encoder = LabelEncoder()
    df['Class_encoded'] = label_encoder.fit_transform(df[target])

    # Select features for the Random Forest model
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',  'Avg_Temp']]

    # Prepare feature and label arrays
    X = features.values
    y = df['Class_encoded'].values

    # Standardize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Set up hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300, 400],  # Number of trees in the forest
        'max_depth': [10, 15, 20, None],      # Maximum depth of the tree
        'min_samples_split': [2, 5, 10, 15],      # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4, 5]     # Minimum samples required to be at a leaf node
    }

    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)

    # Use GridSearchCV for cross-validation and hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=None, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from the cross-validation
    best_model = grid_search.best_estimator_

    # Save the best hyperparameters to a file
    best_params = grid_search.best_params_
    with open(hyperparam_save_path, 'w') as f:
        json.dump(best_params, f)

    print(f"Best hyperparameters saved to {hyperparam_save_path}")
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_}")

    # Save the trained model, scaler, and label encoder
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open(encoder_save_path, 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    print(f"Model, scaler, and label encoder saved to {model_save_path}, {scaler_save_path}, and {encoder_save_path}")

    # Evaluate the model on the test data
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest Classifier')

    output_dir = 'overheat/training/results/rf'
    os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_plot_path = os.path.join(output_dir, 'confusion_matrix_rf.png')
    plt.savefig(confusion_matrix_plot_path)
    plt.show()

    print(f"Confusion matrix plot saved to {confusion_matrix_plot_path}")

    # Plot accuracy based on hyperparameters from GridSearchCV results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Plot accuracy based on number of trees (n_estimators)
    plt.figure(figsize=(10, 6))
    for depth in param_grid['max_depth']:
        subset = results[results['param_max_depth'] == depth]
        plt.plot(subset['param_n_estimators'], subset['mean_test_score'], label=f'max_depth={depth}')
    
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Accuracy vs. Number of Trees for Different Max Depths')
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot_rf.png')
    plt.savefig(accuracy_plot_path)
    plt.show()

    print(f"Accuracy plot saved to {accuracy_plot_path}")

# Testing function for Random Forest Classifier
def test_random_forest_classifier(test_data_path, model_path, scaler_path, encoder_path, output_dir):
    df = pd.read_excel(test_data_path)
    
    plot_all_feature_distributions(df)
    
    # Compute gradients for temperature sensors and voltage
    for i in range(1, 7):
        temp_col = f'T{i}'
        df[f'{temp_col}_diff'] = df[temp_col].diff()

    df['Voltage_diff'] = df['Voltage'].diff()
    df.fillna(0, inplace=True)

    # Load the trained Random Forest Classifier, scaler, and label encoder using pickle
    with open(model_path, 'rb') as model_file:
        rf_model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(encoder_path, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)

    # Select features for the Random Forest model
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',  'Avg_Temp']]
    f_columns = features.columns
    # Standardize features using the loaded scaler
    X_scaled = scaler.transform(features.values)

    # Predict on the test dataset
    y_pred = rf_model.predict(X_scaled)
    
    data_min = scaler.data_min_
    data_max = scaler.data_max_

    print("Min values: ", data_min)
    print("Max values: ", data_max)

    # Encode the true labels for evaluation
    df['Class_encoded'] = label_encoder.transform(df['Class'])

    # Evaluate the model
    y_test = df['Class_encoded'].values
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest Classifier')

    # output_dir = 'overheat/testing/results/rf'
    os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_plot_path = os.path.join(output_dir, 'confusion_matrix_rf.png')
    plt.savefig(confusion_matrix_plot_path)
    plt.show()

    print("Confusion matrix plot saved to {confusion_matrix_plot_path}")

    # Add predicted labels to the original DataFrame
    df['Predicted_Class'] = label_encoder.inverse_transform(y_pred)

    # Save rows with each predicted class to separate Excel files
    safe_rows = df[df['Predicted_Class'] == 'Safe']
    critical_rows = df[df['Predicted_Class'] == 'Critical']
    thermal_runaway_rows = df[df['Predicted_Class'] == 'Thermal Runaway']

    safe_rows.to_excel(os.path.join(output_dir, 'safe_rows_rf.xlsx'), index=False)
    critical_rows.to_excel(os.path.join(output_dir, 'critical_rows_rf.xlsx'), index=False)
    thermal_runaway_rows.to_excel(os.path.join(output_dir, 'thermal_runaway_rows_rf.xlsx'), index=False)
    
    save_tree_plot(rf_model, f_columns, label_encoder.classes_)

    # Calculate performance metrics from confusion matrix
    accuracy = accuracy_score(df['Class_encoded'], y_pred)
    precision = precision_score(df['Class_encoded'], y_pred, average='weighted')
    recall = recall_score(df['Class_encoded'], y_pred, average='weighted')
    f1 = f1_score(df['Class_encoded'], y_pred, average='weighted')

    # Metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)
    
    print("Classified rows saved to Excel files.")
    
def save_tree_plot(rf_model, f_columns, classes):
    export_graphviz(rf_model.estimators_[0], out_file='tree.dot', feature_names=f_columns, class_names=classes, filled=True, rounded=True, proportion=True, max_depth=3)

    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph).render("tree", format="png", cleanup=True)
    
    plt.figure(figsize=(20, 12))
    plot_tree(rf_model.estimators_[0], feature_names=f_columns, class_names=classes, filled=True, rounded=True, proportion=True, max_depth=3)
    plt.title("Decision Tree from Random Forest")
    plt.savefig('decision_tree.png')
    tree_text = export_text(rf_model.estimators_[0], feature_names=f_columns)
    # print(tree_text)
    plt.show()
    
def plot_all_feature_distributions(df):
    """
    Plots the distribution of all numerical features in the DataFrame and saves them to the specified output folder.
    
    Args:
    df (pd.DataFrame): The input DataFrame containing the data.
    output_folder (str): The folder where the plots will be saved.
    """
    output_folder = 'overheat/feature_distributions'
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all columns in the DataFrame
    for feature in df.columns:
        # Plot the distribution of the specified feature
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

        # Save the plot
        file_name = f'{feature}_distribution.png'
        save_path = os.path.join(output_folder, file_name)
        plt.savefig(save_path)
        plt.close()

        print(f"Distribution plot for {feature} saved to {save_path}")
        
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


# Main script execution
if __name__ == "__main__":
    # File paths for training and testing
    training_data_path = 'overheat/training/new_data.xlsx'
    test_data_path = 'overheat/testing/new_data.xlsx'
    validation_data_path = 'overheat/validation/new_data.xlsx'
    test_output_dir = 'overheat/testing/results/rf'
    validation_output_dir = 'overheat/validation/results/rf'
    
    model_save_path = 'overheat/trained_models/rf/random_forest_classifier_model.pkl'
    scaler_save_path = 'overheat/trained_models/rf/scaler.pkl'
    encoder_save_path = 'overheat/trained_models/rf/label_encoder.pkl'
    hyperparam_save_path = 'overheat/trained_models/rf/best_hyperparameters_rf.json'

    # Train the Random Forest Classifier
    # train_random_forest_classifier(training_data_path, model_save_path, scaler_save_path, encoder_save_path, hyperparam_save_path)

    # Test the Random Forest Classifier
    test_random_forest_classifier(validation_data_path, model_save_path, scaler_save_path, encoder_save_path, validation_output_dir)
