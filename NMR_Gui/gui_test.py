from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5 import uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint
from PyQt_Polarization import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

###Circuit Parameters###
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


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width = 5, height=4, dpi=100):
        fig = Figure(figsize = (width,height),dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas,self).__init__(fig)
        fig.tight_layout()
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        # fig = matplotlib.figure.Figure(figsize=(*args,*args), **kwargs)
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = uic.loadUi('gui.ui',self)
        self.resize(888,600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.threadpool = QtCore.QThreadPool()

        self.canvas = MplCanvas(self, width = 5, height = 4, dpi = 100)
        self.ui.gridLayout.addWidget(self.canvas,2,1,1,1)
        self.reference_plot = None    
        # self.q = queue.Queue(maxsize=20)  
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
          

        self.Inputs = Simulate(Config(circ_params,k_range,function_input,scan_s, ranger, Backgmd, Backreal, Current))

        self.x = self.Inputs.LabviewCalculateXArray()
        self.y = self.Inputs.Lineshape(.5,circ_params,k_range,function_input,scan_s, Backgmd, Backreal, Current, ranger)



        # self.x = list(range(100))  # 100 time points
        # self.y = [randint(0,100) for _ in range(100)]  # 100 data points

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def update_plot_data(self):

        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first
        self.y.append( randint(0,100))  # Add a new random value.

        self.data_line.setData(self.x, self.y) 

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())