import gordata as gd
from gui import Ui_MainWindow
import os
import time
import pickle
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import gc
import logging
root = os.getcwd()

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None, dpi = 50):
        self.fig = Figure(figsize=(6,6), tight_layout=True, dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas,self).__init__(self.fig)
        self.fig.tight_layout()

class Worker(qtc.QRunnable):
    def __init__(self, fn):
        super(Worker, self).__init__()
        self.fn = fn
    @qtc.pyqtSlot()
    def run(self):
        try:
            result = self.fn()
        except Exception as e:
            print(e)

class app_gd(qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global dsp
        dsp = gd.dsp()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.msg = ""
        self.ui.startbutton.clicked.connect(self.collect)
        self.ui.stopbutton.clicked.connect(self.interrupt)
        self.ui.openbttn.clicked.connect(self.getFile)
        self.ui.calibutton.clicked.connect(self.calibration)
        self.ui.linkSensor.clicked.connect(self.linkSens)
        self.ui.linkSensor.setEnabled(False)
        self.ui.calibutton.setEnabled(False)
        self.ui.initbttn.clicked.connect(self.initDevices)
        self.ui.loadbttn.clicked.connect(self.loadTF)
        self.ui.combo_TF.currentIndexChanged.connect(self.plotTF)
        
        self.datacache = []
        self.threadpool = qtc.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        
        self.toolbar = None
        self.canv = MatplotlibCanvas(self)
        #self.ui.vLayout_plot.addWidget(self.canv)
        self.toolbarTF = None
        self.canvTF = MatplotlibCanvas(self)
        #self.ui.vLayout_TF.addWidget(self.canvTF)
    def initDevices(self):
        global daq, fs, dt
        #dn = nog.daq()
        daq = gd.daq()
        fs = 1666
        dt = 1/fs
        
        self.devsens={}
        '''
        try:
            with open(root+'sensors.data', 'rb') as f:
                dn.dev = pickle.load(f)
            print(root+'sensors.data loaded')
        except:
            print('no previous sensor data')
        for _dev in dn.dev:
            self.devsens[str(_dev[0])] = str(_dev[-1])
        '''
        self.loadDevices()
        self.ui.linkSensor.setEnabled(True)
        self.ui.calibutton.setEnabled(True)

    def pull(self):
        for addr in daq.devices_list:
            daq.set_device(addr, daq.devices_config[addr])
            time.sleep(daq.dt)
        daq.save_data(daq.pull_data(durr=float(self.ui.label.text())))
        
        self.ui.startbutton.setEnabled(True)

    def collect(self):
        self.ui.startbutton.setEnabled(False)
        worker = Worker(self.pull)
        self.threadpool.start(worker)

    

    def loadDevices(self):
        try:
            self.ui.comboBox.clear()
        except :
            pass
        for address in daq.devices_list:
            self.devsens[str(address)] = str(daq.devices_config[address][-1])
            self.ui.comboBox.addItem(str(address)+'--'+str(daq.devices_config[address][-1]))
        print(self.devsens)

    def interrupt(self):
        daq.state = 0
    
    def getFile(self):
        """ This function will get the address of the csv file location
            also calls a readData function 
        """
        os.chdir(root)
        try:
            os.chdir('DATA')
        except :
            pass

        self.filename = qtw.QFileDialog.getOpenFileName()[0]
        print("File :", self.filename)
        try:
            self.readData()
        except Exception:
            pass

    def calib(self):
        daq.calibrate(daq.dev[self.ui.comboBox.currentIndex()])

    def calibrate(self):
        workal = Worker(self.calib)
        self.threadpool.start(workal)

    def linkSens(self):
        os.chdir(root)
        try:
            os.chdir('sensors')
        except :
            pass

        self.filename = qtw.QFileDialog.getOpenFileName(directory='home/pi/gordata/sensors')[0]
        print("File :", self.filename)
        ii = self.ui.comboBox.currentIndex()
        daq.dev[ii][-1] = self.filename[25:]
        self.loadDevices()
        with open(root+'sensors.data', 'wb') as f:
            pickle.dump(daq.dev, f)
        os.chdir(root)
        np.save('devsens.npy', self.devsens)


    def readData(self):
        self.datacache = pd.read_csv(self.filename, index_col='t')
        self.updatePlot(self.datacache)
        
    def loadTF(self):
        self.datacache = None
        os.chdir(root)
        try:
            os.chdir('DATA')
        except :
            pass

        self.filename = qtw.QFileDialog.getOpenFileName(directory='home/pi/gordata/DATA')[0]
        print("File :", self.filename)
        self.datacache = pd.read_csv(self.filename, index_col='t')
        try:
            self.ui.combo_TF.clear()
        except Exception as e:
            logging.warning(f"Error: {e}")
            pass
        for item in self.datacache.columns:
            self.ui.combo_TF.addItem(item)
        self.ui.combo_TF.setCurrentIndex(0)
        #self.plotTF()


    def plotTF(self):
        plt.clf()        
        frame = str(self.ui.combo_TF.currentText())
        data = self.datacache[[frame]]
        try:
            self.ui.vLayout_TF.removeWidget(self.canvTF)
            self.ui.hLayout_TF.removeWidget(self.toolbarTF)
            self.toolbarTF = None
            self.canvTF = None
        except Exception as e:
            print('warning =>> '+str(e))
            pass
        self.canvTF = MatplotlibCanvas(self)
        self.toolbarTF = Navi(self.canvTF,self.ui.tab_TF)
        self.ui.vLayout_TF.addWidget(self.canvTF,10) 
        self.ui.hLayout_TF.addWidget(self.toolbarTF)
        self.canvTF.axes.cla()
        t, f, S_db = dsp.spect(df=data, print=False)
        self.canvTF.axes.set_xlabel('Time')
        self.canvTF.axes.set_ylabel('Frequency')
        #self.canvTF.axes.set_title('Time-Frequency - {}'.format(frame))
        try:
            #self.canvTF.axes.pcolormesh(t, f, S_db, shading='gouraud',  cmap='turbo')
            self.canvTF.axes.imshow(np.flip(S_db,axis=0), aspect='auto', cmap='turbo', interpolation='gaussian', extent=[t[0], t[-1], f[0], f[-1]])
        except Exception as e:
            print('warning =>> '+str(e))
            pass
        self.canvTF.draw()
        self.canvTF.fig.tight_layout()
       
    def updatePlot(self, plotdata):
        plt.clf()
        try:
            self.ui.vLayout_plot.removeWidget(self.canv)
            self.ui.hLayout_plot.removeWidget(self.toolbar)
            self.toolbar = None
            self.canv = None
        except Exception as e:
            print('warning =>> '+e)
            pass
        self.canv = MatplotlibCanvas(self)
        self.toolbar = Navi(self.canv,self.ui.tab_plot)
        self.ui.hLayout_plot.addWidget(self.toolbar)
        self.ui.vLayout_plot.addWidget(self.canv)
        self.canv.axes.cla()
            
        try:        
                       
            self.canv.axes.plot(plotdata)
            self.canv.axes.legend(plotdata.columns)            
            

        except Exception as e:
            print('==>',e)
        self.canv.draw()

    def showmessage(self, msg):
        msgBox = qtw.QMessageBox()
        msgBox.setIcon(qtw.QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Calibration")
        msgBox.setStandardButtons(qtw.QMessageBox.Ok | qtw.QMessageBox.Cancel)
        
        return msgBox.exec()



    def calibration(self):
        os.chdir(root)
        if 'sensors' not in os.listdir():
            os.mkdir('sensors')
        os.chdir('sensors')
        device = daq.devices_config[daq.devices_list[self.ui.comboBox.currentIndex()]]
        
        

        msg, ok = qtw.QInputDialog().getText(self,
                                        'Name your IMU',
                                        'Type the name of your IMU for calibration: ',
                                        qtw.QLineEdit.Normal)
        if ok and msg:
            sensor ={'name': msg} 
            _path = 'rawdata_{}.csv'.format(sensor['name'])
        else:
                print('cancelled')
                return
        
        NS, ok = qtw.QInputDialog().getInt(self,    'Sample Length',
                                                    'Number seconds per Position: ',
                                                    5, 1, 10, 1)
        if ok:
            self.NS = NS*daq.fs
            print(self.NS)
        else:
                print('cancelled')
                return

        ND, ok = qtw.QInputDialog().getInt(self,  'Sample Length', 
                                                    'Number seconds per Rotation: ',
                                                    5, 1, 10,1)
        if ok:
            self.ND = ND*daq.fs
            print(self.ND)
        else:
                print('cancelled')
                return

        self.calibrationdata = np.zeros((6*self.NS+3*self.ND, 6))
        ii=0
        i=0
        while ii < 6:
            ok = self.showmessage('Move your IMU to the '+str(ii+1)+' position')
            if ok:
                print('collecting position  '+ str(ii+1))   
                    
                ti = tf = time.perf_counter()
                
                while i<(ii+1)*self.NS:
                    tf=time.perf_counter()
                    if tf-ti>=daq.dt:
                        ti = tf
                        try:
                            self.calibrationdata[i,:] = np.array(daq.pull_data(durr=daq.dt, devices=device))
                            i+=1
                        except Exception as e:
                            print(e)
                            self.calibrationdata[i,:] = 6*(0,)
                            
            else:
                print('cancelled')
                return  
            ii+=1
            
            
        print(i)
                
        ii=0    
        while ii <3:
            ok = self.showmessage('Rotate Cube Around Axis '+str(ii+1))
            if ok:        
                   
                print('collecting rotation  '+ str(ii+1))   
                ti = tf = time.perf_counter()
                while i<(6*self.NS+((ii+1)*self.ND)):
                    tf=time.perf_counter()
                    if tf-ti>=daq.dt:
                        ti = tf
                        try:
                            self.calibrationdata[i,:] = np.array(daq.pull_data(durr=daq.dt, devices=device))
                            i+=1
                        except Exception as e:
                            print(e)
                            self.calibrationdata[i,:] = 6*(0,)
                            
            else:
                print('cancelled')
                return  
            ii+=1
        
                                    

        self.calibrationdata = np.array(self.calibrationdata)
        df = pd.DataFrame(self.calibrationdata)
        df.to_csv(_path, index=False)
        #self.updatePlot(self.calibrationdata)
        sensor['acc_p'], sensor['gyr_p'] = daq.calibrate_imu(acc=self.calibrationdata[0:6*self.NS,3:6],
                                                            gyr=self.calibrationdata[:,0:3],
                                                            Ts=self.NS*daq.fs,
                                                            Td=self.ND*daq.fs,
                                                            fs=daq.fs)
                                                            
        #sensor['acc_p'],  = daq.calibacc(self.calibrationdata[0:6*self.NS,3:6], self.NS)
        #sensor['gyr_p'] = daq.calibgyr(self.calibrationdata[:,0:3], self.NS, self.ND) 
        sensorframe = pd.DataFrame(sensor, columns=[ 'acc_p', 'gyr_p'])
        sensorframe.to_csv('{}.csv'.format(sensor['name']))     
        np.savez(sensor['name'], sensor['gyr_p'], sensor['acc_p'])
        gc.collect()

            



if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = app_gd()
    widget.show()
    app.exec_()