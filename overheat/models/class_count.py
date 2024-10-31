import pandas as pd

def limit_decimal_places(df, columns, decimal_places=2):
    """
    Limits the decimal places of specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of columns to round.
    decimal_places (int): Number of decimal places to round to (default is 2).

    Returns:
    pd.DataFrame: DataFrame with the specified columns rounded to the given decimal places.
    """
    df[columns] = df[columns].round(decimal_places)
    return df

# Example usage
def process_and_save_rounded_data(file_path, output_path):
    """
    Reads an Excel file, rounds specified columns to 2 decimal places, and saves the result to a new Excel file.

    Parameters:
    file_path (str): The path to the input Excel file.
    output_path (str): The path where the output Excel file will be saved.

    Returns:
    None
    """
    # Load your data from Excel
    df = pd.read_excel(file_path)

    # Specify the columns to round
    columns_to_round = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Voltage']

    # Apply the rounding function
    df = limit_decimal_places(df, columns_to_round, decimal_places=2)

    # Save the rounded DataFrame to a new Excel file
    df.to_excel(output_path, index=False)  # Save without row index

    print(f"Data with limited decimal places saved to: {output_path}")




def read_excel_and_show_columns():
    """
    This function reads an Excel file and displays the value counts of the 'Class' column.

    It assumes the column 'Class' exists. If it doesn't, it will display the available columns.
    """
    training_data_path = 'overheat/training/data.xlsx'
    test_data_path = 'overheat/testing/data.xlsx'
    val_data_path = 'overheat/validation/data.xlsx'
    
    # Read the Excel file (reads the first sheet by default)
    df = pd.read_excel(val_data_path)
    
    # Show all columns to confirm the column names
    print("Columns in the DataFrame:")
    print(df.columns)
    
    # Specify the expected class column name
    class_name = 'Class'
    
    # Check if the 'Class' column exists
    if class_name in df.columns:
        # Display value counts for the 'Class' column
        print(f"Value counts for '{class_name}' column:")
        print(df[class_name].value_counts())
    else:
        print(f"Error: '{class_name}' column not found. Available columns: {df.columns}")

# read_excel_and_show_columns()
# Example of how to call the function with your data path
training_input_file_path = 'overheat/training/training_data.xlsx'  # Update with your actual input file path
tr_output_file_path = 'overheat/training/data.xlsx'  # The new output file path
test_input_file_path = 'overheat/testing/testing_data.xlsx'  # Update with your actual input file path
test_output_file_path = 'overheat/testing/data.xlsx' 
val_input_file_path = 'overheat/validation/validation_data.xlsx'  # Update with your actual input file path
val_output_file_path = 'overheat/validation/data.xlsx'
# Process the data and save the result
process_and_save_rounded_data(val_input_file_path, val_output_file_path)