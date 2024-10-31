import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Single dataset path
dataset_path = 'overheat/training/data.xlsx'

# Output path for saving simulated images
output_base_folder = 'overheat/simulated_images'

# Step 1: Read the single dataset
data = pd.read_excel(dataset_path)

# Ensure the dataset has the expected columns: Power, T1, T2, T3, T4, T5, T6
required_columns = ['Power (W)', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The dataset is missing one or more required columns: {required_columns}")

# Step 2: Group data by the 'Power' column
grouped_data = data.groupby('Power (W)')

# Step 3: Process each power level separately
for power, group in grouped_data:
    # Step 4: Create an output folder for simulated images for this power level
    output_folder = os.path.join(output_base_folder, f"{int(power)}w")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 5: Average temperature data across all rows (time points) for this power level
    avg_temperatures = group[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].mean()

    # Create a vertical heatmap to simulate temperature distribution
    temp_matrix = np.tile(avg_temperatures.values[:, np.newaxis], (1, 100))  # 100 columns for smoothness

    # Plotting the 2D heatmap
    fig, ax = plt.subplots(figsize=(4, 10))  # Adjust size for a tall heatmap

    # Create the heatmap with dynamic color scaling and flip it vertically
    cax = ax.imshow(temp_matrix, cmap='plasma', aspect='auto', origin='lower',
                    vmin=avg_temperatures.min(), vmax=avg_temperatures.max())

    # Add colorbar to show temperature representation
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

    # Remove axis ticks for a cleaner image
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the image in the corresponding folder
    image_filename = os.path.join(output_folder, f'{int(power)}w_heatmap.png')
    plt.savefig(image_filename, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"Simulated heatmap for {int(power)}w saved to {output_folder}.")
