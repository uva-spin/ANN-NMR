import sys
import pandas as pd

def csv_to_parquet(csv_path, parquet_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # Write to Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {csv_path} to {parquet_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ParquetConverter.py <input.csv> <output.parquet>")
        sys.exit(1)
    csv_path = sys.argv[1]
    parquet_path = sys.argv[2]
    csv_to_parquet(csv_path, parquet_path)
