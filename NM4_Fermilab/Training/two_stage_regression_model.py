import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

data = pd.read_csv('Sample_1.csv')

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values * 100
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values * 100
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values * 100

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


def multi_scale_conv_block(x, filters):
    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    conv7 = layers.Conv1D(filters, 7, padding='same', activation='relu')(x)
    concat = layers.Concatenate()([conv3, conv5, conv7])
    # concat = layers.BatchNormalization()(concat)
    return concat

def attention_block(x):
    squeeze = layers.GlobalAveragePooling1D()(x)
    excitation = layers.Dense(x.shape[-1] // 2, activation='relu')(squeeze)
    excitation = layers.Dense(x.shape[-1], activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, x.shape[-1]))(excitation)
    # excitation = layers.BatchNormalization()(excitation)
    return layers.Multiply()([x, excitation])

def residual_block(x, filters):
    shortcut = x
    conv = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv = layers.Conv1D(filters, 3, padding='same')(conv)
    conv = layers.LayerNormalization()(conv)
    return layers.Add()([shortcut, conv])

def Polarization(input_shape=(500, 1)):
    inputs = Input(shape=input_shape)
    x = multi_scale_conv_block(inputs, 32)
    x = residual_block(x, x.shape[-1])
    x = attention_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.BatchNormalization()(x)

    classifier = layers.Dense(16, activation='relu')(x)
    is_low_P = layers.Dense(1, activation='sigmoid')(classifier)


    ### Let's introduce a temperature parameter to the sigmoid function to smooth the transition
    temperature = 3.0
    is_low_P = layers.Activation(lambda z: tf.sigmoid(z * temperature), name='classifier')(is_low_P)

    low_reg = layers.Dense(32, activation='relu')(x)
    low_reg = layers.Dense(1, name='reg_low')(low_reg)

    high_reg = layers.Dense(32, activation='relu')(x)
    high_reg = layers.Dense(1, name='reg_high')(high_reg)

    output = layers.Lambda(lambda tensors: tensors[0] * tensors[1] + (1 - tensors[0]) * tensors[2], name='P_output')([is_low_P, low_reg, high_reg])

    model = models.Model(inputs=inputs, outputs=[is_low_P, output])
    return model

def compile_and_train(model, X_train, y_train, epochs=50, batch_size=32):
    y_class = (y_train < 10.0).astype(int) ### True/False if P is less than whatever percentage
    y_reg = y_train.astype('float32')

    model.compile(
        optimizer='adam',
        loss={'classifier': 'binary_crossentropy', 'P_output': 'mse'},
        loss_weights={'classifier': 0.2, 'P_output': 1.0},
        metrics={'P_output': 'mae'}
    )

    history = model.fit(
        X_train, {'classifier': y_class, 'P_output': y_reg},
        validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1
    )
    return history

def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # classifier loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['classifier_loss'], label='Classifier Loss')
    plt.plot(history.history['val_classifier_loss'], label='Val Classifier Loss')
    plt.title('Classifier Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['P_output_mae'], label='Regression MAE')
    plt.plot(history.history['val_P_output_mae'], label='Val Regression MAE')
    plt.title('Regression MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_errors(model, X_test, y_test):

    _, y_pred = model.predict(X_test)
    
    relative_errors = ((y_pred - y_test) / y_test) * 100
    absolute_errors = np.abs(y_pred - y_test)
    mae = np.mean(absolute_errors)
    
    eval_df = pd.DataFrame({
        'True_P': y_test.flatten(),
        'Predicted_P': y_pred.flatten(),
        'Absolute_Error': absolute_errors.flatten(),
        'Relative_Percent_Error': relative_errors.flatten()
    })
    
    eval_df.to_csv('model_evaluation_metrics_higher.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, relative_errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    
    mean_error = np.mean(relative_errors)
    plt.axhline(y=mean_error, color='g', linestyle='--', label=f'Mean Error: {mean_error:.2f}%')
    
    plt.xlabel('True Polarization (P)')
    plt.ylabel('Relative Percent Error (%)')
    plt.title('Distribution of Relative Percent Errors vs True Polarization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution_higher.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(relative_errors, bins=50, alpha=0.7)
    plt.axvline(x=mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.2f}%')
    plt.xlabel('Relative Percent Error (%)')
    plt.ylabel('Count')
    plt.title('Histogram of Relative Percent Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_histogram_higher.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean Relative Error: {mean_error:.2f}%")
    print(f"Median Relative Error: {np.median(relative_errors):.2f}%")
    print(f"Standard Deviation of Errors: {np.std(relative_errors):.2f}%")
    print(f"Maximum Error: {np.max(np.abs(relative_errors)):.2f}%")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    with open('evaluation_statistics_higher.txt', 'w') as f:
        f.write(f"Mean Relative Error: {mean_error:.2f}%\n")
        f.write(f"Median Relative Error: {np.median(relative_errors):.2f}%\n")
        f.write(f"Standard Deviation of Errors: {np.std(relative_errors):.2f}%\n")
        f.write(f"Maximum Error: {np.max(np.abs(relative_errors)):.2f}%\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
    
    return relative_errors, eval_df

model = Polarization()
history = compile_and_train(model, X_train, y_train)
plot_history(history)

relative_errors, eval_df = evaluate_and_plot_errors(model, X_test, y_test)


