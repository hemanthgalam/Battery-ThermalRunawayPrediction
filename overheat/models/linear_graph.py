import pandas as pd
import os
import matplotlib.pyplot as plt


sample_file_path = 'overheat/training/data.xlsx'
output_dir = 'overheat/results/training'

data = pd.read_excel(sample_file_path)
temp_columns = ['T1','T2','T3','T4','T5','T6']
voltage_column = ['Voltage']
power_levels = [20, 30, 50, 70, 90]

for power in power_levels:
    folder_name = f'{output_dir}/{power}W'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    power_data = data[data['Power (W)'] == power]
    
    for voltage in voltage_column:
        plt.plot(power_data['Time'], power_data[voltage], label=voltage)

    plt.title(f'Voltage vs Time for Power = {power}w')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (volts)')
    plt.legend()

    plot_filename = os.path.join(folder_name, f'voltage_{power}W_plot.png')
    plt.savefig(plot_filename)
    plt.clf()

print("Plots saved successfully in respective folders.")

for power in power_levels:
    folder_name = f'{output_dir}/{power}W'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    power_data = data[data['Power (W)'] == power]
    
    for temp in temp_columns:
        plt.plot(power_data['Time'], power_data[temp], label=temp)
    
    plt.title(f'Temperature vs Time for Power = {power}w')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    plot_filename = os.path.join(folder_name, f'power_{power}W_plot.png')
    plt.savefig(plot_filename)
    
    plt.clf()

print("Plots saved successfully in respective folders.")