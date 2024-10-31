import pandas as pd
import numpy as np
import os


folder_paths = {
    '20w': 'overheat/data/20w',
    '30w': 'overheat/data/30w',
    '50w': 'overheat/data/50w',
    '70w': 'overheat/data/70w',
    '90w': 'overheat/data/90w'
}

power_levels = {
    '20w': 20,
    '30w': 30,
    '50w': 50,
    '70w': 70,
    '90w': 90
}

training_output_dir = 'overheat/training'
testing_output_dir = 'overheat/testing'
validation_output_dir = 'overheat/validation'

training_output_file = os.path.join(training_output_dir, 'regression_data.xlsx')
testing_output_file = os.path.join(testing_output_dir, 'regression_data.xlsx')
validation_output_file = os.path.join(validation_output_dir, 'regression_data.xlsx')


def compute_gradients_by_power(df):
    # Group by 'Power' and compute the difference for each group, keeping the original order
    # grouped = df.groupby('Power (W)')

    # Compute temperature gradients for T1 to T6 within each power level
    for i in range(1, 7):
        temp_col = f'T{i}'
        df[f'{temp_col}_diff'] = df[temp_col].diff().fillna(0)  # Fill first row within each power group with 0

    # Compute voltage gradient within each power level
    df['Voltage_diff'] = df['Voltage'].diff().fillna(0)  # Fill first row within each power group with 0

    return df

# Adding the average temperature function
def add_avg_temperature(df):
    df['Avg_Temp'] = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].mean(axis=1)
    return df

# Function to filter temperatures less than 300
def filter_temperatures(df):
    temperature_columns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    df = df[(df[temperature_columns] < 200).all(axis=1)]
    return df

# Function to downsample a DataFrame by selecting every nth row
def downsample_data(df, step_size):
    return df.iloc[::step_size].reset_index(drop=True)

def process_data(file_name, folder_paths, power_levels, output_file):
    data = []
    
    for folder_name, folder_path in folder_paths.items():
        power = power_levels[folder_name]

        if not os.path.exists(folder_path) or power is None:
            continue

        for run_file in os.listdir(folder_path):
            if run_file.endswith('.xlsx') and run_file == file_name:
                run_file_path = os.path.join(folder_path, run_file)

                # Read Excel data with proper numeric conversions
                df = pd.read_excel(run_file_path, converters={
                    'Time': lambda x: str(x).replace(',', '.'),  # Handle commas
                    'Voltage': lambda x: str(x).replace(',', '.'),
                    'T1': lambda x: str(x).replace(',', '.'),
                    'T2': lambda x: str(x).replace(',', '.'),
                    'T3': lambda x: str(x).replace(',', '.'),
                    'T4': lambda x: str(x).replace(',', '.'),
                    'T5': lambda x: str(x).replace(',', '.'),
                    'T6': lambda x: str(x).replace(',', '.')
                })

                # Convert necessary columns to numeric types
                df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
                df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
                df['T1'] = pd.to_numeric(df['T1'], errors='coerce')
                df['T2'] = pd.to_numeric(df['T2'], errors='coerce')
                df['T3'] = pd.to_numeric(df['T3'], errors='coerce')
                df['T4'] = pd.to_numeric(df['T4'], errors='coerce')
                df['T5'] = pd.to_numeric(df['T5'], errors='coerce')
                df['T6'] = pd.to_numeric(df['T6'], errors='coerce')

                df = df.dropna(subset=['Voltage'])
                df[['Voltage']] = df[['Voltage']].clip(lower=0)
            
                # Drop unnecessary columns if they exist
                df = df.drop(columns=['ShuntVoltage', 'ShuntCurrent'], errors='ignore')

                # Ensure all temperature columns have valid values
                temperature_columns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
                df[temperature_columns] = df[temperature_columns].clip(lower=0)
                df = filter_temperatures(df)
                df = compute_gradients_by_power(df)
                df = add_avg_temperature(df)
       
                
                data.append(df)
    
    final_df = pd.concat(data, ignore_index=True)

    # Save the processed data to an Excel file
    final_df.to_excel(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

# Function to process training data
def process_training_data():
    if not os.path.exists(training_output_dir):
        os.makedirs(training_output_dir)

    process_data('Run2.xlsx', folder_paths, power_levels, training_output_file)

# Function to process testing data
def process_testing_data():
    if not os.path.exists(testing_output_dir):
        os.makedirs(testing_output_dir)

    process_data('Run1.xlsx', folder_paths, power_levels, testing_output_file)
    
def process_validation_data():
    if not os.path.exists(validation_output_dir):
        os.makedirs(validation_output_dir)

    process_data('Run3.xlsx', folder_paths, power_levels, validation_output_file)

# Run the processing functions
process_training_data()
process_testing_data()
process_validation_data()
