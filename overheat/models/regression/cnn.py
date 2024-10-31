import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras._tf_keras.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the data
def load_data(data_path):
    df = pd.read_excel(data_path)
    df['Max_Temperature'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1)
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Avg_Temp',
                   'T1_diff', 'T2_diff', 'T3_diff', 'T4_diff', 'T5_diff', 'T6_diff', 'Voltage_diff']]
    X = features.values
    y = df['Max_Temperature'].values
    return X, y

# Prepare the data for CNN + LSTM
def prepare_data(X, y, time_steps=10):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped, y_reshaped = [], []
    for i in range(len(X_scaled) - time_steps):
        X_reshaped.append(X_scaled[i:i+time_steps])
        y_reshaped.append(y[i+time_steps])
    return np.array(X_reshaped), np.array(y_reshaped), scaler

# Build the CNN + LSTM model with enhancements
def build_cnn_lstm_model(input_shape):
    model = tf.keras.Sequential()
    # First CNN Layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))

    # Second CNN Layer
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu',  padding='same'))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))

    # LSTM Layer
    model.add(tf.keras.layers.LSTM(units=200, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))

    # Dense Layers for Regression
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mean_squared_error')
    return model

# Train the model
def train_model(X_train, y_train, X_val, y_val, model_save_path):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val))
    model.save(model_save_path)
    return model, history

# Test the model using test data set and save results
def test_model(test_data_path, model_path, scaler_path, time_steps=10):
    model = tf.keras.models.load_model(model_path)
    plot_model(model, to_file='CNN_LSTM_regression_model_architecture.png', show_shapes=True)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    X_test, y_test = load_test_data(test_data_path, scaler, time_steps)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test R^2: {r2}")
    print(f"Test MAE: {mae}")

    results = {"Test MSE": mse, "Test R^2": r2, "Test MAE": mae}
    output_dir = 'overheat/testing/results/cnn_regression'
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.5)
    plt.scatter(y_test, y_test, color='red', label='Actual', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'b--', label='Perfect Prediction')
    plt.xlabel('Actual Max Temperature')
    plt.ylabel('Predicted Max Temperature')
    plt.title('Actual vs Predicted Max Temperature (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_cnn_lstm_testing.png'))
    plt.show()

# Load and preprocess the test data
def load_test_data(data_path, scaler, time_steps=10):
    df = pd.read_excel(data_path)
    df['Max_Temperature'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1)
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Avg_Temp',
                   'T1_diff', 'T2_diff', 'T3_diff', 'T4_diff', 'T5_diff', 'T6_diff', 'Voltage_diff']]
    X = features.values
    y = df['Max_Temperature'].values
    X_scaled = scaler.transform(X)
    X_reshaped, y_reshaped = [], []
    for i in range(len(X_scaled) - time_steps):
        X_reshaped.append(X_scaled[i:i+time_steps])
        y_reshaped.append(y[i+time_steps])
    return np.array(X_reshaped), np.array(y_reshaped)

# Main script execution
if __name__ == "__main__":
    training_data_path = 'overheat/training/regression_data.xlsx'
    test_data_path = 'overheat/testing/regression_data.xlsx'
    model_save_path = 'overheat/trained_models/cnn_lstm_model.h5'
    scaler_save_path = 'overheat/trained_models/cnn_lstm_scaler.pkl'
    
    X, y = load_data(training_data_path)
    # X_reshaped, y_reshaped, scaler = prepare_data(X, y, time_steps=10)
    # X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.3, random_state=42)
    # model, history = train_model(X_train, y_train, X_val, y_val, model_save_path)
    
    # with open(scaler_save_path, 'wb') as scaler_file:
    #     pickle.dump(scaler, scaler_file)
    
    test_model(test_data_path, model_save_path, scaler_save_path)
