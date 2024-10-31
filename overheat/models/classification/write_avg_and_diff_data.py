import pandas as pd
import numpy as np
import os

training_output_dir = 'overheat/training'
testing_output_dir = 'overheat/testing'
validation_output_dir = 'overheat/validation'

training_output_file = os.path.join(training_output_dir, 'regression_data.xlsx')
testing_output_file = os.path.join(testing_output_dir, 'regression_data.xlsx')
validation_output_file = os.path.join(validation_output_dir, 'regression_data.xlsx')

training_data_path = 'overheat/training/data.xlsx'
test_data_path = 'overheat/testing/data.xlsx'
validation_data_path = 'overheat/validation/data.xlsx'

def compute_gradients_by_power(df):
    # Group by 'Power' and compute the difference for each group, keeping the original order
    grouped = df.groupby('Power (W)')

    # Compute temperature gradients for T1 to T6 within each power level
    for i in range(1, 7):
        temp_col = f'T{i}'
        df[f'{temp_col}_diff'] = grouped[temp_col].diff().fillna(0)  # Fill first row within each power group with 0

    # Compute voltage gradient within each power level
    df['Voltage_diff'] = grouped['Voltage'].diff().fillna(0)  # Fill first row within each power group with 0

    return df

# Adding the average temperature function
def add_avg_temperature(df):
    df['Avg_Temp'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].mean(axis=1)
    return df

# Process the data by adding gradients and average temperature
def process_data(file_path, output_file):
    # Load the data
    df = load_data(file_path)

    # Add average temperature
    df = add_avg_temperature(df)

    # Compute gradients for temperature sensors and voltage based on power levels
    df = compute_gradients_by_power(df)

    # Ensure the original order is maintained
    df = df.sort_index()

    # Save the processed data to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

# Functions for loading data and handling paths
def load_data(file_path):
    return pd.read_excel(file_path)

# Create directories and process training, testing, and validation data
def process_training_data():
    if not os.path.exists(training_output_dir):
        os.makedirs(training_output_dir)
    process_data(training_data_path, training_output_file)

def process_testing_data():
    if not os.path.exists(testing_output_dir):
        os.makedirs(testing_output_dir)
    process_data(test_data_path, testing_output_file)

def process_validation_data():
    if not os.path.exists(validation_output_dir):
        os.makedirs(validation_output_dir)
    process_data(validation_data_path, validation_output_file)

# Run the processing functions
process_training_data()
process_testing_data()
process_validation_data()
