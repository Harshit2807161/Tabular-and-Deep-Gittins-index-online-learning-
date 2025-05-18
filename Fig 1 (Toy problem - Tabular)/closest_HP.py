import pandas as pd

# Read the CSV file
df = pd.read_csv('QGI_HP_data_final_toy_refined.csv')

print(df.columns)

df_filtered = df[df['phi'] != 0]

# Given list of true M values
true_M_values = [0.9, 0.819, 0.74804, 0.6909, 0.64565]

# Function to calculate the sum of absolute differences
def sum_absolute_differences(row, true_values):
    return sum(abs(row[f'M[{i}]'] - true_values[i]) for i in range(len(true_values)))

# Apply the function to each row in the filtered DataFrame and create a new column with the result
df_filtered['sum_abs_diff'] = df_filtered.apply(lambda row: sum_absolute_differences(row, true_M_values), axis=1)

# Find the row with the smallest sum of absolute differences
closest_row = df_filtered.loc[df_filtered['sum_abs_diff'].idxmin()]

# Extract the hyperparameters
qlr = closest_row['qlr']
wlr = closest_row['wlr']
phi = closest_row['phi']
den = closest_row['den']

# Print the result
print(f'The hyperparameters corresponding to the closest M values are:')
print(f'qlr: {qlr}, wlr: {wlr}, phi: {phi}, den: {den}')
