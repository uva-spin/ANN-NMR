import numpy as np
import tensorflow as tf


def Predict_P_and_Error(Polarization_Model_Path, Error_Model_Path, Array):

    ### Load in models outside of function. This improves speed when implemented in LabVIEW

    Polarization_Model = tf.keras.models.load_model(Polarization_Model_Path)
    Error_Model = tf.keras.models.load_model(Error_Model_Path)

    ### Predicts Polarization, Relative Error, and Absolute Error. Defined in function in order to implement in LabVIEW.

    ### Returns
    # Y_Pol: Predicted Polarization
    # Y_Rel: Predicted Relative Error
    # Y_Abs: Predicted Absolute Error

    X = np.array(Array) ## Data input 

    ### Predicted Polarization
    Y_Pol = Polarization_Model.predict(X)
    Y_Pol = Y_Pol.reshape((len(Y_Pol),))

    ### Predicted Errors
    Y_Rel,Y_Abs = Error_Model.predict(X)
    Y_Rel = Y_Rel.reshape((len(Y_Rel),))
    Y_Abs = Y_Abs.reshape((len(Y_Abs),))

    return Y_Pol, Y_Rel, Y_Abs

