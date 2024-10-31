import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras._tf_keras.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the data
def load_data(data_path):
    df = pd.read_excel(data_path)
    
    # Define the target variable
    df['Max_Temperature'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1)

    # Select the features and the target variable
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Avg_Temp',
                   'T1_diff', 'T2_diff', 'T3_diff', 'T4_diff', 'T5_diff', 'T6_diff', 'Voltage_diff']]
    X = features.values
    y = df['Max_Temperature'].values
    
    return X, y

# Prepare the data for CNN + LSTM
def prepare_data(X, y, time_steps=10):
    # Standardize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape X into 3D array for CNN + LSTM (samples, time_steps, features)
    X_reshaped = []
    y_reshaped = []
    
    for i in range(len(X_scaled) - time_steps):
        X_reshaped.append(X_scaled[i:i+time_steps])
        y_reshaped.append(y[i+time_steps])

    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)
    
    return X_reshaped, y_reshaped, scaler

# Build the CNN + LSTM model
def build_cnn_lstm_model(input_shape):
    model = tf.keras.Sequential()
    # CNN Layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape))
    
    # LSTM Layer
    model.add(tf.keras.layers.LSTM(units=100, return_sequences=False))
    # Dense Layers for Regression
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))  # Output layer for regression
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
    
    return model

# Train the model
def train_model(X_train, y_train, X_val, y_val, model_save_path):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the model
    model.save(model_save_path)
    
    return model, history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the model using test data set and save results
def test_model(test_data_path, model_path, scaler_path, time_steps=10):
    # Load the scaler and trained model
    model = tf.keras.models.load_model(model_path)
    plot_model(model, to_file='CNN_LSTM_model_architecture.png', show_shapes=True)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Load and preprocess the test data
    X_test, y_test = load_test_data(test_data_path, scaler, time_steps)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Compute regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test R^2: {r2}")
    print(f"Test MAE: {mae}")

    # Save the results to a dictionary
    results = {
        "Test MSE": mse,
        "Test R^2": r2,
        "Test MAE": mae
    }

    # Save the results to a JSON file
    output_dir = 'overheat/testing/results/cnn_lstm'
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # Plot predicted vs actual values with different colors
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.5)
    plt.scatter(y_test, y_test, color='red', label='Actual', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Max Temperature')
    plt.ylabel('Predicted Max Temperature')
    plt.title('Actual vs Predicted Max Temperature (Test Set)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_cnn_lstm_testing.png'))
    plt.show()

# Load and preprocess the test data
def load_test_data(data_path, scaler, time_steps=10):
    df = pd.read_excel(data_path)

    # Define the target variable
    df['Max_Temperature'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1)

    # Select the features and the target variable
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Avg_Temp',
                   'T1_diff', 'T2_diff', 'T3_diff', 'T4_diff', 'T5_diff', 'T6_diff', 'Voltage_diff']]
    X = features.values
    y = df['Max_Temperature'].values

    # Standardize features using the same scaler used in training
    X_scaled = scaler.transform(X)

    # Reshape X into 3D array for CNN + LSTM (samples, time_steps, features)
    X_reshaped = []
    y_reshaped = []

    for i in range(len(X_scaled) - time_steps):
        X_reshaped.append(X_scaled[i:i+time_steps])
        y_reshaped.append(y[i+time_steps])

    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)
    
    return X_reshaped, y_reshaped

# Main script execution
if __name__ == "__main__":
    # File paths for training and testing data
    training_data_path = 'overheat/training/regression_data.xlsx'
    test_data_path = 'overheat/testing/regression_data.xlsx'
    
    model_save_path = 'overheat/trained_models/cnn_lstm_model.h5'
    scaler_save_path = 'overheat/trained_models/cnn_lstm_scaler.pkl'
    
    # Load and prepare the data
    X, y = load_data(training_data_path)
    X_reshaped, y_reshaped, scaler = prepare_data(X, y, time_steps=10)

    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.3, random_state=42)
    # 
    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val, model_save_path)

    # Save the scaler
    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Plot training history
    # plot_training_history(history)

    # Test the CNN + LSTM regression model using the test dataset
    test_model(test_data_path, model_save_path, scaler_save_path)
