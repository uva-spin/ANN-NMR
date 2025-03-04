import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=';')
    print("Data columns:", data.columns)
    print(data.head()) 
    return data

def plot_loss(data):
    if 'epoch' not in data.columns or 'loss' not in data.columns:
        raise ValueError("The CSV does not have the required 'epoch' and 'loss' columns.")

    plt.figure(figsize=(8, 6))

    plt.plot(data['loss'], color='blue', linestyle='-', label = 'Loss')
    plt.plot(data['val_loss'], color='red', linestyle='-', label = 'Validation Loss')


    plt.title('Loss Values Over Epochs For ND3 Model', fontsize=16, weight='bold')
    plt.xlabel('Epoch', fontsize=16, weight='bold')
    plt.ylabel('Loss', fontsize=16, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True)

    plt.tight_layout()
    plt.legend()
    plt.savefig('ND3_SGD_Loss_Plot.png', dpi=600)  
    plt.show()

if __name__ == "__main__":
    file_path = r'training_log_Deuteron_v14_SGD.csv'  
    data = load_data(file_path)
    plot_loss(data)

