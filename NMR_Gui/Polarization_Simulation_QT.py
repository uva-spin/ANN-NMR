import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ReadCSV import read_latest_row_in_real_time
from Predict import Predict


class LivePlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(LivePlotCanvas, self).__init__(fig)
        self.setParent(parent)
        self.plot(np.zeros(500), np.linspace(-3, 3, 500))  # Adjusted x_array to cover -3 to 3

    def plot(self, data, x_array):
        self.axes.clear()
        self.axes.set_title('Real-time Prediction Plot')
        self.axes.set_xlabel('R')
        self.axes.set_ylabel('Intensity')
        self.axes.set_xlim([-3, 3])
        self.axes.grid(True)
        self.axes.plot(x_array, data, label='Data', color='blue')  # Plot data
        self.axes.legend()

        # Draw plot
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.data = np.zeros((500, 1))  # Initialize with zeros
        self.prediction_value = 0.0  # Initialize predicted value
        self.csv_file = ''  # Initialize as empty
        self.output_csv = 'output_data.csv'  # CSV file to save output
        self.output_df = None  # DataFrame to store output

        self.initUI()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second

    def initUI(self):
        self.setWindowTitle('Real-time Data Prediction')
        self.setGeometry(100, 100, 800, 600)  # Set the window size

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Plot Canvas
        self.plot_canvas = LivePlotCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.plot_canvas)

        # Prediction Label Box
        self.prediction_label = QLabel('Prediction: N/A')
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 16px; font-weight: bold; border: 2px solid black; padding: 8px;")
        layout.addWidget(self.prediction_label)

        # Button Layout (CSV and Model Selection)
        button_layout = QHBoxLayout()

        self.file_button = QPushButton('Select CSV File')
        self.file_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.file_button.clicked.connect(self.select_csv_file)
        button_layout.addWidget(self.file_button)

        self.model_button = QPushButton('Select Model File')
        self.model_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.model_button.clicked.connect(self.select_model_file)
        button_layout.addWidget(self.model_button)

        layout.addLayout(button_layout)

        # Spacer for Aesthetics
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        central_widget.setLayout(layout)

    def select_csv_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.csv_file = file_name

    def select_model_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.h5);;All Files (*)", options=options)
        if file_name:
            self.model = tf.keras.models.load_model(file_name)

    def update_plot(self):
        if not self.csv_file or not self.model:
            return  # If no file is selected or model is not loaded, do nothing

        try:
            # Read data from CSV file
            data_array = read_latest_row_in_real_time(self.csv_file).reshape(500, 1)

            if data_array.size == 500:
                self.data = data_array

                # Predict using the pre-trained model
                prediction = Predict(self.data)  # Call the Predict function
                self.prediction_value = float(prediction)  # Assuming prediction is a scalar

                # Update plot
                x_array = np.linspace(-3, 3, 500)
                self.plot_canvas.plot(self.data, x_array)

                # Update predicted value label
                self.prediction_label.setText(f'Prediction: {self.prediction_value:.4f}')

                # Save to CSV
                self.save_to_csv()

            else:
                print(f"Received data size mismatch: expected 500, got {data_array.size}")

        except Exception as e:
            print(f"Error reading data: {e}")

    def save_to_csv(self):
        try:
            if self.output_df is None:
                # Initialize DataFrame if not already initialized
                self.output_df = pd.DataFrame(columns=['Data', 'Prediction'])

            # Append new row to DataFrame
            new_row = {'Data': np.squeeze(self.data), 'Prediction': self.prediction_value}
            self.output_df = self.output_df.append(new_row, ignore_index=True)

            # Save DataFrame to CSV
            self.output_df.to_csv(self.output_csv, index=False)
            print(f"Saved data to {self.output_csv}")

        except Exception as e:
            print(f"Error saving data to CSV: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
