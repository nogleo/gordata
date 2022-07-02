import logging
import gc
import pandas as pd
import numpy as np
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
import gordata as gd
from gui import Ui_MainWindow
import os

root = os.getcwd()

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=80):
        fig = Figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        


class Worker(qtc.QRunnable):
    def __init__(self, fn):
        super(Worker, self).__init__()
        self.fn = fn

    @qtc.pyqtSlot()
    def run(self):
        try:
            result = self.fn()
        except Exception as e:
            logging.debug("couldn`t run", exc_info=e)


class app_gd(qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.basicConfig(level=logging.INFO)
        global dq, dsp
        dq = gd.daq()
        dsp = gd.dsp()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.msg: str = ""
        self.ui.pButton_start.clicked.connect(self.collect)
        self.ui.pButton_stop.clicked.connect(self.stop_collect)
        self.ui.pButton_load_viz.clicked.connect(self.load_viz)
        self.ui.pButton_calib.clicked.connect(self.calibration)
        self.ui.pButthon_link.clicked.connect(self.linkSens)
        self.ui.pButthon_link.setEnabled(False)
        self.ui.pButton_calib.setEnabled(False)
        self.ui.pButton_scan.clicked.connect(self.initDevices)
        self.ui.cBox_data.currentIndexChanged.connect(self.updatePlot)
        self.ui.cBox_method.currentIndexChanged.connect(self.updatePlot)
        self.data_buffer: pd.DataFrame = pd.DataFrame()
        self.datacache = []
        self.threadpool = qtc.QThreadPool()
        logging.debug("Multithreading with maximum %d threads" %
                      self.threadpool.maxThreadCount())
        self.canv = MatplotlibCanvas(self)
        self.ui.vLayout_viz.addWidget(self.canv)
        self.navigation = Navi(self.canv, self.ui.tab_viz)
        self.ui.hLayout_viz.addWidget(self.navigation)
        
        
    def initDevices(self):
        self.device_list = {}
        self.loadDevices()
        self.ui.pButthon_link.setEnabled(True)
        self.ui.pButton_calib.setEnabled(True)

    def pull(self):
        logging.info('Time to collect: '+self.ui.label_durr.text())
        dq.session = self.ui.line_session.text()
        logging.info('main pull')
        dq.pull_data(durr=float(self.ui.label_durr.text()))   
        logging.info('pull data ok')
        self.ui.pButton_start.setEnabled(True)

    def collect(self):
        self.ui.pButton_start.setEnabled(False)
        worker = Worker(self.pull)
        self.threadpool.start(worker)

    def loadDevices(self):
        try:
            self.ui.listWidget.clear()
        except Exception as e:
            logging.warning("can`t clear comobo box", exc_info=e)
            pass
        
        for address, device in dq.devices.items():
            item = qtw.QListWidgetItem('{} ---- {}'.format(address,device['cal']))
            self.ui.listWidget.addItem(item)
            
            
            logging.debug(f"Device {address} loaded")
        self.ui.listWidget.setCurrentItem(item)

    def interrupt(self):
        dq.running = False
        logging.info('set dq.running to False')

    def stop_collect(self):
        worker = Worker(self.interrupt)
        self.threadpool.start(worker)

    def linkSens(self):
        self.filename = qtw.QFileDialog.getOpenFileName(
            directory='home/pi/gordata/sensors')[0]
        logging.info("File :", self.filename)
        addr = self.ui.listWidget.currentItem().text()[:3]
        dq.devices[int(addr)]['cal'] = self.filename[25:-4]
        self.loadDevices()
        
    def load_viz(self):
        del self.datacache
        self.ui.cBox_method.setCurrentIndex(0)
        
        files = qtw.QFileDialog.getOpenFileNames(directory='home/pi/gordata/data')[0]
        DF =  [pd.read_csv(file, index_col='t') for file in files]
        self.datacache = pd.concat(DF)

        try:
            self.ui.cBox_data.clear()
        except Exception as e:
            logging.warning("can`t clear combo box", exc_info=e)
            pass
        for item in self.datacache.columns:
            self.ui.cBox_data.addItem(item)
        self.ui.cBox_data.setCurrentIndex(0)

    
    def updatePlot(self):   
        plt.clf()     
        try:
            self.canv.axes.clear()      
            #self.ui.vLayout_viz.removeWidget(self.canv)
        except Exception as e:
            logging.info('warning =>> ', exc_info=e)
            pass
        #self.canv = MatplotlibCanvas(self)  
        #self.ui.vLayout_viz.addWidget(self.canv)
        
        frame = self.ui.cBox_data.currentText()
        if self.ui.cBox_method.currentText() == 'Time':
            self.canv.axes.set_xlabel('Time')
            self.canv.axes.set_ylabel('Amplitude')
            try:
                self.canv.axes.plot(self.datacache)
                self.canv.axes.legend(self.datacache.columns)
            except Exception as e:
                logging.debug('Can`t plot data ==>', exc_info=e)
        elif self.ui.cBox_method.currentText() == 'PSD':
            self.canv.axes.set_xlabel('Frequency')
            self.canv.axes.set_ylabel('Amplitude')
            try:
              f, S_db = dsp.PSD(self.datacache, fs=dq.fs, return_fig=False)
              self.canv.axes.plot(f, S_db)
              self.canv.axes.legend(self.datacache.columns)
            except Exception as e:
                logging.warning('can`t plot PSD',exc_info=e)
        elif self.ui.cBox_method.currentText() == 'STFT':
            self.canv.axes.set_xlabel('Time')
            self.canv.axes.set_ylabel('Frequency')
            try:
                t, f, S_db = dsp.spect(df=self.datacache[[frame]], print=False)
                self.canv.axes.imshow(S_db, aspect='auto', cmap='turbo',
                                    interpolation='gaussian', extent=[t[0], t[-1], f[0], f[-1]])
            except Exception as e:
                logging.warning('can`t plot STFT',exc_info=e)
        elif self.ui.cBox_method.currentText() == 'WSST':
            self.canv.axes.set_xlabel('Time')
            self.canv.axes.set_ylabel('Frequency')
            plt.yscale('log')
            
            try:
                t, f, S_db = dsp.WSST(df=self.datacache[frame],fs=dq.fs)
                self.canv.axes.imshow(S_db, aspect='auto', cmap='turbo',
                                    interpolation='gaussian',
                                    extent=[t[0], t[-1], f[0], f[-1]])
            except Exception as e:
                logging.warning('can`t plot WSST',exc_info=e)
        plt.yscale('linear')
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
        addr = int(self.ui.listWidget.currentItem().text()[:3])
        device = {addr: dq.devices[addr]}
        msg, ok = qtw.QInputDialog().getText(self,
                                             'Name your IMU',
                                             'Type the name of your IMU for calibration: ',
                                             qtw.QLineEdit.Normal)
        if ok and msg:
            _name = msg
            _path = 'rawdata_{}.csv'.format(_name)
        else:
            logging.debug('cancelled')
            return
        TS, ok = qtw.QInputDialog().getInt(self,
                                           'Sample Length',
                                           'Number seconds per Position: ',
                                           5, 1, 10, 1)
        if ok:
            self.NS = TS*dq.fs
            logging.debug(self.NS)
        else:
            logging.debug('cancelled')
            return
        TD, ok = qtw.QInputDialog().getInt(self,
                                           'Sample Length',
                                           'Number seconds per Rotation: ',
                                           5, 1, 10, 1)
        if ok:
            self.ND = TD*dq.fs
            logging.debug(self.ND)
        else:
            logging.debug('cancelled')
            return

        self.calibrationdata = np.zeros((6*self.NS+3*self.ND, 6))
        
        ii = 0
        
        while ii < 6:
            ok = self.showmessage(
                'Move your IMU to the '+str(ii+1)+' position')
            if ok:
                logging.debug('collecting position  ' + str(ii+1))
                try:
                    self.calibrationdata[ii*self.NS:(ii+1)*self.NS,:] = dq.pull_data(durr=TS, devices=device, rtrn_array=True)
                except Exception as e:
                    logging.warning('can`t pull data', exc_info=e)
            else:
                logging.debug('cancelled')
                return False
            ii += 1


        ii = 0
        while ii < 3:
            ok = self.showmessage('Rotate Cube Around Axis '+str(ii+1))
            if ok:
                logging.info('collecting rotation  ' + str(ii+1))
                
                self.calibrationdata[6*self.NS+ii*self.ND:6*self.NS+(ii+1)*self.ND] = dq.pull_data(durr=TD, devices=device, rtrn_array=True)
            else:
                logging.info('cancelled')
                return False
            ii += 1
        pd.DataFrame(self.calibrationdata,
                     columns=['Gx','Gy','Gz','Ax','Ay','Az']).to_csv(_path)

        acc_p, gyr_p = dq.calibrate_imu(acc=self.calibrationdata[:, 3:6],
                                        gyr=self.calibrationdata[:, 0:3],
                                        Ts=TS, Td=TD, fs=dq.fs, name=_name)

        pd.DataFrame({'acc_p':acc_p, 'gyr_p':gyr_p}).to_csv(dq.root+'/sensors/{}.csv'.format(_name))
        
        logging.info("Garbage collection: {}".format(gc.collect()))


if __name__ == '__main__':
    app = qtw.QApplication([])
    widget = app_gd()
    widget.show()
    app.exec_()
