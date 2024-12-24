import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from Misc_Functions import *
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(f"Error setting GPU: {e}")


tf.keras.mixed_precision.set_global_policy('mixed_float16')


data_path = find_file("Deuteron_V8_2_100_No_Noise_500K.csv")
version = 'Deuteron_2_100_v16'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def Polarization(): 
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(500,))) 

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=256, activation='relu6', kernel_regularizer=regularizers.L2(1e-2)
    ))
    model.add(tf.keras.layers.Dropout(0.2))  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=128, activation='relu6', kernel_regularizer=regularizers.L2(1e-2)
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=64, activation='relu6', kernel_regularizer=regularizers.L2(1e-2)
    ))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))  

    model.compile(
            optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss='mse',
        metrics=['mse']
    )

    return model



callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=0, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-10),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min')
]


data = pd.read_csv(data_path)

val_fraction = 0.2
test_fraction = 0.1

train_split_index = int(len(data) * (1 - val_fraction - test_fraction))
val_split_index = int(len(data) * (1 - test_fraction))

train_data = data.iloc[:train_split_index]
val_data = data.iloc[train_split_index:val_split_index]
test_data = data.iloc[val_split_index:]

target_variable = "P"
X_train, y_train = train_data.drop([target_variable, 'SNR'], axis=1).values, train_data[target_variable].values
X_val, y_val = val_data.drop([target_variable, 'SNR'], axis=1).values, val_data[target_variable].values
X_test, y_test = test_data.drop([target_variable, 'SNR'], axis=1).values, test_data[target_variable].values


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Polarization()

print("Starting training on full dataset...")
history = model.fit(
    X_train,
    y_train, 
    validation_data=(X_val, y_val), 
    epochs=200,
    batch_size=32,
    callbacks=callbacks_list
)


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

loss_plot_path = os.path.join(performance_dir, f'{version}_Loss_Plot.png')
plt.savefig(loss_plot_path, dpi=600)

print(f"Loss plot saved to {loss_plot_path}")


model_summary_path = os.path.join(performance_dir, 'model_summary.txt')
with open(model_summary_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

model.save(os.path.join(model_dir, f'final_model_{version}.keras'))

print("Evaluating on test data...")

test_loss, test_mse = model.evaluate(X_test, y_test, batch_size=16)

y_test_pred = model.predict(X_test)
residuals = y_test - y_test_pred.flatten()

test_results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred.flatten(),
    'Residuals': residuals
})

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

plt.figure(figsize=(10, 6))

residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)

fig = plt.figure(figsize=(16, 6))  

gs = fig.add_gridspec(1, 2)  

ax1 = fig.add_subplot(gs[0])

plot_histogram(
    residuals*100, 
    'Histogram of Polarization Difference', 
    'Difference in Polarization', 
    'Count', 
    'red', 
    ax1,
    plot_norm=False
)
ax2 = fig.add_subplot(gs[1])
plot_histogram(
    np.abs(residuals*100), 
    'Histogram of Mean Absolute Error', 
    'Mean Absolute Error',
    '', 
    'orange', 
    ax2,
    plot_norm=False
)

ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, 
         ha='center', fontsize=16,weight='bold')
ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, 
         ha='center', fontsize=16,weight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

output_path = os.path.join(performance_dir, f'{version}_Histograms.png')
fig.savefig(output_path,dpi=600)

print(f"Histograms plotted in {output_path}!")

test_summary_results = {
    'Date': [str(datetime.now())],
    'Test Loss': [test_loss],
    'Test MSE': [test_mse]
}

summary_results_df = pd.DataFrame(test_summary_results)

summary_results_file = os.path.join(performance_dir, f'test_summary_results_{version}.csv')
summary_results_df.to_csv(summary_results_file, index=False)

print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")
print(f"Test summary results saved to {summary_results_file}")
print(f"Model summary saved to {model_summary_path}")