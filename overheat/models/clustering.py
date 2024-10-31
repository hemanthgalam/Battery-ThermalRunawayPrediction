import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np

# Define folder paths and power levels
folder_paths = {
    '20w': 'overheat/data/20w',
    '30w': 'overheat/data/30w',
    '31w': 'overheat/data/30w_vertical',
    '50w': 'overheat/data/50w',
    '70w': 'overheat/data/70w',
    '90w': 'overheat/data/90w'
}

power_levels = {
    '20w': 20,
    '30w': 30,
    '31w': 31,
    '50w': 50,
    '70w': 70,
    '90w': 90
}

# Output folder for clustering results
clustering_output_dir = 'overheat/clustering'
clustering_output_file = os.path.join(clustering_output_dir, 'clustering_results.xlsx')
plot_output_file = os.path.join(clustering_output_dir, 'clustering_plot.png')

# Function to perform K-Means clustering
def determine_kmeans_clusters(df, n_clusters=3):
    max_temperatures = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1).values.reshape(-1, 1)

    # Fit KMeans clustering model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(max_temperatures)
    
    return kmeans

# Function to process the data and perform clustering
def process_data_for_clustering_and_limits(file_name, folder_paths, power_levels, output_file):
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
                    'Time': lambda x: str(x).replace(',', '.'),
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

                # Append processed DataFrame to data list
                data.append(df)

    # Combine all data into one DataFrame
    final_df = pd.concat(data, ignore_index=True)

    # Determine clusters using K-Means
    kmeans_model = determine_kmeans_clusters(final_df)
    
    # Apply clustering labels to each row
    final_df['Cluster'] = kmeans_model.predict(final_df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max(axis=1).values.reshape(-1, 1))

    # Group by cluster and calculate min and max for each cluster
    cluster_groups = final_df.groupby('Cluster')[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].max().max(axis=1)
    cluster_limits = cluster_groups.sort_values().values  # Sorted max values for each cluster

    print(f"Cluster boundaries (max temperatures per cluster): {cluster_limits}")

    # Derive boundaries between clusters (midpoint between max of one cluster and min of next)
    limits = np.mean([cluster_limits[:-1], cluster_limits[1:]], axis=0)
    print(f"Derived temperature limits between clusters: {limits}")

    # Save the clustered data to an Excel file for analysis
    final_df.to_excel(output_file, index=False)
    print(f"Clustering results saved to {output_file}")

# Function to process clustering data and save plot
def process_clustering_data_with_limits():
    if not os.path.exists(clustering_output_dir):
        os.makedirs(clustering_output_dir)

    process_data_for_clustering_and_limits('Run1.xlsx', folder_paths, power_levels, clustering_output_file)

# Run the clustering processing and save the plot
process_clustering_data_with_limits()
