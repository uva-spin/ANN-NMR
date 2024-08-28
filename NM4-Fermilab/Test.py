import os
import datetime
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers, callbacks
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm.keras import TqdmCallback
from keras.activations import *
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


# Enable logging to check GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

# Check for available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

if tf.config.list_physical_devices('GPU'):
    print("GPU is available and recognized by TensorFlow!")
else:
    print("No GPU detected. Please ensure that your GPU and drivers are properly configured.")