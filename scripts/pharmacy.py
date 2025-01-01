import pandas as pd
import os
# Define the directory containing your datasets
data = r"C:\Users\Hasan\Desktop\week4"

# File names
files = ["store.csv", "test.csv", "train.csv"]

# Read and print datasets
for file in files:
    file_path = os.path.join(data, file)
    df = pd.read_csv(file_path)  # Read the CSV file
    print(f"\nContents of {file}:")
    print(df.head())  # Print the first few rows of the dataset