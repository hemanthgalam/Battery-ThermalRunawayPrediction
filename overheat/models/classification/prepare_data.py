import pandas as pd
import numpy as np
import os

# Define folder paths and power levels
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

training_output_file = os.path.join(training_output_dir, 'heat_data.xlsx')
testing_output_file = os.path.join(testing_output_dir, 'heat_data.xlsx')
validation_output_file = os.path.join(validation_output_dir, 'heat_data.xlsx')

# Function to classify the stage based on max temperature
def classify_stage(row):
    max_temp = row[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max()
    if max_temp < 60:
        return 'Safe'
    elif max_temp < 120:
        return 'Critical'
    else:
        return 'Thermal Runaway'

# Function to downsample a DataFrame by selecting every nth row
def downsample_data(df, step_size):
    return df.iloc[::step_size].reset_index(drop=True)

# Function to downsample the data based on temperature and power level
def downsample_based_on_temperature(df, power):
    # Downsample for different temperature classes
    high_temp_df = df[df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) > 120]
    high_temp_downsampled = high_temp_df.iloc[::20]  # Thermal Runaway: Keep every 8th row for 2-second interval
    
    critical_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) >= 60) & 
                          (df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 120)]
    critical_temp_downsampled = critical_temp_df.iloc[::1]  # Critical: Keep every 7th row
    safe_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 60)]

    # Safe class with power level of 20W: Downsample to 5 ms (example: keep every 6th row if the interval is 30 ms)
    if power == 30:
        safe_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 60) & (df['Class'] == 'Safe')]
        safe_temp_downsampled = downsample_data(safe_temp_df, step_size=1)  # Adjust the step size for 5 ms interval
        critical_temp_downsampled = downsample_data(critical_temp_downsampled, step_size=3)
        # high_temp_downsampled = downsample_data(critical_temp_downsampled, step_size=1)
    elif power == 50:
        safe_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 60) & (df['Class'] == 'Safe')]
        safe_temp_downsampled = downsample_data(safe_temp_df, step_size=15)
        critical_temp_downsampled = downsample_data(critical_temp_downsampled, step_size=2)
        # high_temp_downsampled = downsample_data(critical_temp_downsampled, step_size=1)
    elif power == 70:
        safe_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 60) & (df['Class'] == 'Safe')]
        safe_temp_downsampled = downsample_data(safe_temp_df, step_size=1)
        critical_temp_downsampled = downsample_data(critical_temp_downsampled, step_size=5)
        high_temp_downsampled = downsample_data(high_temp_downsampled, step_size=7)
    elif power == 90:
        safe_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 60) & (df['Class'] == 'Safe')]
        safe_temp_downsampled = downsample_data(safe_temp_df, step_size=1)
        critical_temp_downsampled = downsample_data(critical_temp_downsampled, step_size=7)
    else:
        safe_temp_df = df[(df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1) < 60)]
    
        safe_temp_downsampled = safe_temp_df  # No downsampling for other power levels

    # Concatenate downsampled data and return the final DataFrame
    return pd.concat([high_temp_downsampled, critical_temp_downsampled, safe_temp_downsampled]).sort_values(by='Time').reset_index(drop=True)

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

                # Drop rows with missing voltage data and filter out negative voltages
                df = df.dropna(subset=['Voltage'])
                df[['Voltage']] = df[['Voltage']].clip(lower=0)
            
                
                # Drop unnecessary columns if they exist
                df = df.drop(columns=['ShuntVoltage', 'ShuntCurrent'], errors='ignore')

                # Ensure all temperature columns have valid values
                temperature_columns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
                df[temperature_columns] = df[temperature_columns].clip(lower=0)

                df['Power (W)'] = power

                # Classify based on temperature before downsampling
                df['Class'] = df.apply(classify_stage, axis=1)

                # Downsample the data based on temperature and power level
                # df = downsample_based_on_temperature(df, power)

                # Append processed DataFrame to data list
                data.append(df)

    # Combine all data into one DataFrame
    final_df = pd.concat(data, ignore_index=True)

    # Print the class distribution
    class_counts = final_df['Class'].value_counts()
    print("\nClass distribution:")
    print(class_counts)

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
# process_training_data()
process_testing_data()
# process_validation_data()
