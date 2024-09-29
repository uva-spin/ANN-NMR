import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file using pandas with semicolon as the delimiter
def load_data(file_path):
    # Load the CSV with the correct delimiter
    data = pd.read_csv(file_path, delimiter=';')
    print("Data columns:", data.columns)
    print(data.head())  # Print first few rows to ensure it's loaded correctly
    return data

# Plot the loss values using matplotlib
def plot_loss(data):
    if 'epoch' not in data.columns or 'loss' not in data.columns:
        raise ValueError("The CSV does not have the required 'epoch' and 'loss' columns.")

    plt.figure(figsize=(8, 6))

    # Plot the loss values
    plt.plot(data['epoch'], data['loss'], color='blue', linestyle='-')
    plt.plot(data['epoch'], data['val_loss'], color='red', linestyle='-')


    # Customize the plot for publication
    plt.title('Loss Values Over Epochs For NH3 Model', fontsize=16, weight='bold')
    plt.xlabel('Epoch', fontsize=16, weight='bold')
    plt.ylabel('Loss', fontsize=16, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid for better readability
    plt.grid(True)

    # Save the figure with high resolution
    plt.tight_layout()
    plt.savefig('Proton_Loss_Plot.png', dpi=600)  # Save as a high-resolution PNG file
    plt.show()

# Main function
if __name__ == "__main__":
    file_path = r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NM4_Fermilab\Model Performance\v9\training_log_v9.csv'  
    data = load_data(file_path)
    plot_loss(data)

