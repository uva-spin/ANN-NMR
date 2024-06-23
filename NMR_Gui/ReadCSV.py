import pandas as pd
import time
import os

def read_latest_row_in_real_time(file_path, interval=1):
    """
    Reads the latest row of a CSV file in real time, updating at a specified interval.

    :param file_path: Path to the CSV file.
    :param interval: Time interval (in seconds) at which to check for updates.
    :return: The latest row of the CSV file as a pandas Series.
    """
    last_row_count = 0

    while True:
        try:
            # Get the current number of rows in the file
            current_row_count = sum(1 for _ in open(file_path)) - 1  # Subtracting 1 for header row
            
            # Check if the file has new rows since the last read
            if current_row_count > last_row_count:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Get the latest row
                latest_row = df.iloc[-1]
                print("Latest row:")
                print(latest_row.to_dict())
                
                # Update the last row count
                last_row_count = current_row_count
                
                return latest_row
            
            # Wait for the specified interval before checking again
            time.sleep(interval)
        
        except FileNotFoundError:
            print(f"File {file_path} not found. Waiting for the file to be created...")
            time.sleep(interval)
        except pd.errors.EmptyDataError:
            print("CSV file is empty. Waiting for data to be written...")
            time.sleep(interval)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(interval)

