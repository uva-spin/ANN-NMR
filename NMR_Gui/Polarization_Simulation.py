import dearpygui.dearpygui as dpg
import math
import time
import threading
from PyQt_Polarization import *

###Circuit Parameters###
P = .5
U = 0.1
# Cknob = 0.125
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
Backgmd = np.loadtxt(r'C:\Work\ANN\ANN-NMR\NMR_Gui\Backgmd.dat', unpack = True)
Backreal = np.loadtxt(r'C:\Work\ANN\ANN-NMR\NMR_Gui\Backreal.dat', unpack = True)
Current = np.loadtxt(r'C:\Work\ANN\ANN-NMR\NMR_Gui\New_Current.csv', unpack = True)

nsamples = 500

global data_y
global data_x
# Can use collections if you only need the last 100 samples
# data_y = collections.deque([0.0, 0.0],maxlen=nsamples)
# data_x = collections.deque([0.0, 0.0],maxlen=nsamples)

# Use a list if you need all the data. 
# Empty list of nsamples should exist at the beginning.
# Theres a cleaner way to do this probably.
data_y = [0.0] * nsamples
data_x = [0.0] * nsamples

class Parameters():
        # Cknob = 0.125
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

def Update_Parameters_Callback(sender, value):

    Parameters.P = value

    dpg.set_value(sender, Parameters.P)

def update_data():
    Inputs = Simulate(Config(circ_params,k_range,function_input,scan_s, ranger, Backgmd, Backreal, Current))
    while True:

        data_x = Inputs.LabviewCalculateXArray()
        data_y = Inputs.Lineshape(Parameters.P,Parameters.circ_params,Parameters.k_range,Parameters.function_input,Parameters.scan_s, Backgmd, Backreal, Current, Parameters.ranger)
        
        #set the series x and y to the last nsamples
        dpg.set_value('series_tag', [list(data_x), list(data_y)])          
        dpg.fit_axis_data('x_axis')
        dpg.fit_axis_data('y_axis')
        
        # time.sleep(13/200)
        # sample=sample+1
           


dpg.create_context()
with dpg.window(label='NMR Simulation', tag='win',width=800, height=600):
    # P = dpg.add_input_float(label="P")
    # U = dpg.add_input_float(label="U")
    # Cknob = dpg.add_input_float(label="Cknob")
    # cable = dpg.add_input_float(label="Cable Length")
    # eta = dpg.add_input_float(label="Eta")
    # phi = dpg.add_input_float(label="Phi")
    # Cstray = dpg.add_input_float(label="CStray")
    # k_range = dpg.add_input_float(label="K Range")

    # function_input = dpg.add_input_float(label="F Input")
    # scan_s = dpg.add_input_float(label="Scan Sweep")
    # ranger = dpg.add_input_float(label="Range")

    # circ_params = (U,Cknob,cable,eta,phi,Cstray)

    dpg.add_input_float(label="P", callback=Update_Parameters_Callback, user_data=Parameters.P, default_value=P)
    dpg.add_input_float(label="U", callback=Update_Parameters_Callback, user_data=Parameters.U, default_value=U)
    dpg.add_input_float(label="Cknob", callback=Update_Parameters_Callback, user_data=Parameters.Cknob, default_value=Cknob)
    dpg.add_input_float(label="Cable Length", callback=Update_Parameters_Callback, user_data=Parameters.cable,default_value=cable)
    dpg.add_input_float(label="Eta", callback=Update_Parameters_Callback, user_data=Parameters.eta,default_value=eta)
    dpg.add_input_float(label="Phi", callback=Update_Parameters_Callback, user_data=Parameters.phi,default_value=phi)
    dpg.add_input_float(label="CStray", callback=Update_Parameters_Callback, user_data=Parameters.Cstray,default_value=Cstray)
    dpg.add_input_float(label="K Ranger", callback=Update_Parameters_Callback, user_data=Parameters.k_range,default_value=k_range)
    dpg.add_input_float(label="F Input", callback=Update_Parameters_Callback, user_data=Parameters.function_input,default_value=function_input)
    dpg.add_input_float(label="Scan Sweep", callback=Update_Parameters_Callback, user_data=Parameters.scan_s,default_value=scan_s)
    dpg.add_input_float(label="Range", callback=Update_Parameters_Callback, user_data=Parameters.ranger,default_value=ranger)


    with dpg.plot(label='Deuteron Lineshape', height=240, width=500):
        # optionally create legend
        dpg.add_plot_legend()

        # REQUIRED: create x and y axes, set to auto scale.
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='x_axis')
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='y_axis')


        # series belong to a y axis. Note the tag name is used in the update
        # function update_data
        dpg.add_line_series(x=data_x,y=data_y, 
                            label='Lineshape', parent='y_axis', 
                            tag='series_tag')
        
            
                            
dpg.create_viewport(title='Custom Title', width=850, height=640)

dpg.setup_dearpygui()
dpg.show_viewport()

thread = threading.Thread(target=update_data)
thread.start()
dpg.start_dearpygui()

dpg.destroy_context()