import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras._tf_keras.keras.utils import plot_model
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to compute gradients
def compute_gradients(df):
    for i in range(1, 7):
        temp_col = f'T{i}'
        df[f'{temp_col}_diff'] = df[temp_col].diff()
    
    df['Voltage_diff'] = df['Voltage'].diff()
    df.fillna(0, inplace=True)
    return df

# Function to classify stages based on max temperature
def classify_stage(row):
    max_temp = row[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max()
    if max_temp < 60:
        return 'Safe'
    elif max_temp < 120:
        return 'Critical'
    else:
        return 'Thermal Runaway'

# Function to preprocess the data and add labels
def preprocess_data(df):
    df['Stage'] = df.apply(classify_stage, axis=1)
    label_encoder = LabelEncoder()
    df['Stage_encoded'] = label_encoder.fit_transform(df['Stage'])
    return df, label_encoder

# Function to extract features and labels
def extract_features_and_labels(df):
    features = df[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',  'Avg_Temp']]
    labels = df['Stage_encoded'].values
    return features, labels

# Function to standardize features
def standardize_features(features):
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))
    return features_scaled, scaler

# Function to split data into training and test sets
def split_data(X, y, test_size=0.1, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to build the CNN model
def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1)) 
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model


# Function to compile and train the model
def train_model(model, X_train, y_train, epochs=1000, batch_size=32):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3)
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, label_encoder):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(f"Test Accuracy: {test_acc}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    return y_pred

# Function to save plots and results
def save_plots(history, y_test, y_pred, label_encoder, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save training and validation accuracy plot
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy Over Epochs')
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    plt.show()

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - 1D CNN')
    confusion_matrix_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_plot_path)
    plt.show()

    print(f"Accuracy plot saved to {accuracy_plot_path}")
    print(f"Confusion matrix plot saved to {confusion_matrix_plot_path}")

# Function to save model, scaler, and label encoder
def save_model_and_encoders(model, scaler, label_encoder, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save CNN model
    model_save_path = os.path.join(save_dir, 'cnn_model.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the scaler
    scaler_save_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_save_path}")

    # Save the label encoder
    encoder_save_path = os.path.join(save_dir, 'label_encoder.pkl')
    with open(encoder_save_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {encoder_save_path}")

# Function to save classified rows
def save_classified_rows(df, output_dir):
    safe_rows = df[df['Predicted_Stage'] == 'Safe']
    critical_rows = df[df['Predicted_Stage'] == 'Critical']
    thermal_runaway_rows = df[df['Predicted_Stage'] == 'Thermal Runaway']

    safe_rows.to_excel(os.path.join(output_dir, 'safe_rows.xlsx'), index=False)
    critical_rows.to_excel(os.path.join(output_dir, 'critical_rows.xlsx'), index=False)
    thermal_runaway_rows.to_excel(os.path.join(output_dir, 'thermal_runaway_rows.xlsx'), index=False)

    print("Classified rows saved to Excel files.")



def test_saved_model(test_file_path, model_dir, output_dir):
    # Load the test data
    df_test = pd.read_excel(test_file_path)

    # Compute gradients (same as training)
    df_test = compute_gradients(df_test)

    # Preprocess the test data (same as training)
    # We need to add the 'Stage' classification and then encode it to 'Stage_encoded'
    df_test['Stage'] = df_test.apply(classify_stage, axis=1)  # Classify stage based on temperatures

    # Load label encoder to encode the 'Stage' column
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    df_test['Stage_encoded'] = label_encoder.transform(df_test['Stage'])  # Encode the stages

    # Select features (same as training)
    features_test = df_test[['Time', 'Voltage', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',  'Avg_Temp']]

    # Load model and scaler
    model = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    model.summary()
    
    plot_model(model, to_file='CNN_model_architecture.png', show_shapes=True)
    
    # Scale the features
    X_test_scaled = scaler.transform(features_test)
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # Make predictions
    y_pred = np.argmax(model.predict(X_test_scaled), axis=-1)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    # Add predictions to the DataFrame
    df_test['Predicted_Stage'] = predicted_labels

    # Save test results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_test.to_excel(os.path.join(output_dir, 'test_results.xlsx'), index=False)
    
    # Save confusion matrix plot
    conf_matrix = confusion_matrix(df_test['Stage_encoded'], y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - 1D CNN')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_cnn.png')
    plt.savefig(confusion_matrix_path)
    plt.show()

    # Calculate and plot accuracy
    test_loss, test_acc = model.evaluate(X_test_scaled, df_test['Stage_encoded'], verbose=0)
    print(f"Test Loss: {test_loss}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Calculate performance metrics from confusion matrix
    accuracy = accuracy_score(df_test['Stage_encoded'], y_pred)
    precision = precision_score(df_test['Stage_encoded'], y_pred, average='weighted')
    recall = recall_score(df_test['Stage_encoded'], y_pred, average='weighted')
    f1 = f1_score(df_test['Stage_encoded'], y_pred, average='weighted')

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


    print(f"Test results, accuracy plot, confusion matrix, and metrics saved to {output_dir}")


# Main function to execute the pipeline
def main():
    # Step 1: Train and evaluate model (as done before)
    file_path = 'overheat/training/new_data.xlsx'
    test_file_path = 'overheat/testing/new_data.xlsx'
    val_file_path = 'overheat/validation/new_data.xlsx'
    df = load_data(file_path)
    df = compute_gradients(df)
    df, label_encoder = preprocess_data(df)
    features, labels = extract_features_and_labels(df)
    # features_scaled, scaler = standardize_features(features)
    # X_train, X_test, y_train, y_test = split_data(features_scaled, labels)
    # model = build_model(input_shape=(X_train.shape[1], 1))
    # model, history = train_model(model, X_train, y_train)
    # y_pred = evaluate_model(model, X_test, y_test, label_encoder)

    # # Step 2: Save model, plots, and results
    # output_dir = 'overheat/training/results/cnn'
    # save_plots(history, y_test, y_pred, label_encoder, output_dir)
    save_dir = 'overheat/trained_models/cnn'
    # save_model_and_encoders(model, scaler, label_encoder, save_dir)

    # Step 3: Test saved model on new data
    test_output_dir = 'overheat/testing/results/cnn'
    val_output_dir = 'overheat/validation/results/cnn'
    test_saved_model(val_file_path, save_dir, val_output_dir)

# Run the main function
if __name__ == "__main__":
    main()
