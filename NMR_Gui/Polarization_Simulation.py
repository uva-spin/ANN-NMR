import dearpygui.dearpygui as dpg
from math import *
import time
import threading
import pandas as pd
import numpy as np
from PyQt_Polarization import *
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model

###Circuit Parameters###
P = .5
U = 0.1
Cknob = 0.016
cable = 2.5
eta = 0.0104
phi = 6.1319
Cstray = 10**(-15)
k_range = 5000
circ_params = (U,Cknob,cable,eta,phi,Cstray)
function_input = 32000000
scan_s = .25
ranger = 0
Backgmd = np.loadtxt(r'Backgmd.dat', unpack = True)
Backreal = np.loadtxt(r'Backreal.dat', unpack = True)
Current = np.loadtxt(r'New_Current.csv', unpack = True)

testmodel = tf.keras.models.load_model(r'trained_model_1M_v5.h5')



class Parameters():
    P = P
    U = U
    Cknob = Cknob
    cable = cable
    eta = eta
    phi = phi
    Cstray = Cstray
    k_range = k_range
    circ_params = (U,Cknob,cable,eta,phi,Cstray)
    function_input = function_input
    scan_s = scan_s
    ranger = ranger

class Polarization_Predicted():
    P_Pred = P

if Parameters.ranger == 1:
    nsamples = 5000
else:
    nsamples = 500

global data_y
global data_x
global data_y_pred

data_y = [0.0] * nsamples
data_x = [0.0] * nsamples
result_pred_new = [0.0] * nsamples



p_pred=0

def Update_Parameters_Callback(sender, value):
    setattr(Parameters, sender, value)
    dpg.set_value(sender, value)


def Update_Polarization_Callback(sender, value):
    setattr(Polarization_Predicted, sender, value)
    dpg.set_value(sender, value)


def update_data():
    Inputs = Simulate(Config(Parameters.circ_params,Parameters.k_range,Parameters.function_input,Parameters.scan_s, Parameters.ranger, Backgmd, Backreal, Current))
    while True:

        data_x = Inputs.LabviewCalculateXArray((Parameters.U,Parameters.Cknob,Parameters.cable,Parameters.eta,Parameters.phi,Parameters.Cstray),Parameters.scan_s,Parameters.k_range,Parameters.ranger,Parameters.function_input)
        data_y = Inputs.Lineshape(Parameters.P,(Parameters.U,Parameters.Cknob,Parameters.cable,Parameters.eta,Parameters.phi,Parameters.Cstray),Parameters.function_input,Parameters.scan_s, Parameters.ranger,Parameters.k_range)

        
        p, p_pred, result_new, result_pred_new = Predict(data_y, Parameters.P, testmodel)

        # data_y_pred =Inputs.Lineshape(p_pred,(Parameters.U,Parameters.Cknob,Parameters.cable,Parameters.eta,Parameters.phi,Parameters.Cstray),Parameters.function_input,Parameters.scan_s, Parameters.ranger,Parameters.k_range)\
        
        dpg.set_value('series_tag', [list(data_x), list(data_y)])    
        dpg.set_value('series_tag2', [list(data_x), list(result_pred_new)])        
        dpg.fit_axis_data('x_axis')
        dpg.fit_axis_data('y_axis')
        dpg.fit_axis_data('x_axis2')
        dpg.fit_axis_data('y_axis2')
        
        # time.sleep(13/200)
           


dpg.create_context()
with dpg.window(label='NMR Simulation', tag='win',width=800, height=600):


    circ_params = (U,Cknob,cable,eta,phi,Cstray)

    dpg.add_input_float(tag="P", label = "P", callback=Update_Parameters_Callback, default_value=P)
    dpg.add_input_float(tag="U",label="U", callback=Update_Parameters_Callback, default_value=U)
    dpg.add_input_float(tag="Cknob",label="Cknob", callback=Update_Parameters_Callback, default_value=Cknob)
    dpg.add_input_float(tag="cable",label="cable", callback=Update_Parameters_Callback ,default_value=cable)
    dpg.add_input_float(tag="eta",label="eta", callback=Update_Parameters_Callback,default_value=eta)
    dpg.add_input_float(tag="phi",label="phi", callback=Update_Parameters_Callback,default_value=phi)
    dpg.add_input_float(tag="Cstray",label="Cstray", callback=Update_Parameters_Callback ,default_value=Cstray)
    dpg.add_input_float(tag="k_range",label="k_range", callback=Update_Parameters_Callback,default_value=k_range)
    dpg.add_input_float(tag="function_input",label="function_input", callback=Update_Parameters_Callback,default_value=function_input)
    dpg.add_input_float(tag="scan_s",label="scan_s", callback=Update_Parameters_Callback,default_value=scan_s)
    dpg.add_input_int(tag="ranger",label="ranger", callback=Update_Parameters_Callback,default_value=ranger)

    dpg.add_float_value(tag = "P_Pred", label = "Predicted P", callback = Update_Polarization_Callback,default_value=0.5)

    with dpg.plot(label='Deuteron Lineshape', height=240, width=500):
        dpg.add_plot_legend()

    
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='x_axis')
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='y_axis')

        dpg.add_line_series(x=data_x,y=data_y, 
                            label='Lineshape', parent='y_axis', 
                            tag='series_tag')
        
    with dpg.plot(label='Filtered Signal', height=240, width=500):
        # optionally create legend
        dpg.add_plot_legend()

        # REQUIRED: create x and y axes
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='x_axis2')
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='y_axis2')

        # series belong to a y axis
        dpg.add_line_series(x=data_x,y=result_pred_new, 
                            label='Lineshape', parent='y_axis2', 
                            tag='series_tag2')
        
        
dpg.create_viewport(title='Custom Title', width=850, height=640)

dpg.setup_dearpygui()
dpg.show_viewport()

thread = threading.Thread(target=update_data)
thread.start()
dpg.start_dearpygui()

dpg.destroy_context()