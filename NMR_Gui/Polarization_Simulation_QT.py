import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import socket
from Lineshape_Code_V2 import *

# Example lineshape function (replace with your actual function)
def produce_lineshape(prediction):
    # Implement your actual lineshape generation logic here
    return prediction * np.sin(np.linspace(0, 2 * np.pi, 500))

class LivePlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(LivePlotCanvas, self).__init__(fig)
        self.setParent(parent)
        self.plot()

    def plot(self, data, lineshape):
        self.axes.clear()
        self.axes.set_title('Real-time Prediction Plot')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Value')
        self.axes.grid(True)
        self.axes.plot(data, label='Data')
        self.axes.plot(lineshape, label='Lineshape', linestyle='--')
        self.axes.legend()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self, model, host, port):
        super().__init__()

        self.model = model
        self.data = np.zeros((500, 1))  # Initialize with zeros
        self.predictions = np.zeros((500, 1))  # Initialize with zeros
        self.lineshape = np.zeros((500, 1))  # Initialize with zeros
        self.prediction_value = 0.0  # Initialize predicted value

        self.initUI()
        
        # Setup socket connection to LabVIEW
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))  # Connect to LabVIEW server

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second

    def initUI(self):
        self.setWindowTitle('Real-time Data Prediction')

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.plot_canvas = LivePlotCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.plot_canvas)

        self.prediction_label = QLabel('Prediction: N/A')
        layout.addWidget(self.prediction_label)

    def update_plot(self):
        try:
            # Read data from LabVIEW
            data_size = 500  # Number of data points expected
            received_data = self.socket.recv(data_size * 10).decode('utf-8')  # Adjust buffer size if needed

            # Assuming data is sent as a comma-separated string
            data_array = np.fromstring(received_data, sep=',').reshape(data_size, 1)

            if data_array.size == data_size:
                self.data = data_array

                # Predict using the pre-trained model
                prediction = self.model.predict(self.data.reshape(1, -1))
                self.predictions = np.roll(self.predictions, -1)
                self.predictions[-1] = prediction

                # Generate lineshape using the prediction
                self.lineshape = produce_lineshape(prediction)

                # Update plot
                self.plot_canvas.plot(self.data, self.lineshape)

                # Update predicted value label
                self.prediction_value = float(prediction)  # Assuming prediction is a scalar
                self.prediction_label.setText(f'Prediction: {self.prediction_value:.4f}')
                
            else:
                print(f"Received data size mismatch: expected {data_size}, got {data_array.size}")

        except Exception as e:
            print(f"Error receiving data: {e}")

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    model_path = 'path_to_your_model.h5'
    model = load_model(model_path)

    # Replace with your LabVIEW server IP and port
    labview_host = 'localhost'
    labview_port = 65432

    main_window = MainWindow(model, labview_host, labview_port)
    main_window.show()

    sys.exit(app.exec_())
