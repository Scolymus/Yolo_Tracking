# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

# Form implementation generated from reading ui file 'train.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject, QProcess, pyqtSignal, pyqtSlot
from train_yolo.GUI_train_image import GUI_train_image
import os
import subprocess
import shlex
import sys
import ctypes 
import threading
import shutil

class GUI_train(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        uic.loadUi(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"GUI"+os.path.sep+'train.ui', self)

        self.btn_browse_dataset.clicked.connect(self.browse_folder_data)
        self.btn_browse_yolo.clicked.connect(self.browse_folder_yolo)
        self.btn_start.clicked.connect(self.train)

        self.train_image_gui = GUI_train_image()
        self.train_image_gui.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowMinimizeButtonHint)
    
        self.worker = Worker()
        self.worker.outSignal.connect(self.darknet2GUI)
        self.worker.outEnd.connect(self.command_finnished)
        self.worker.outAux.connect(self.auxiliar)

    def browse_folder_data(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select dataset directory')
        self.txt_dataset.setText(folder)

    def browse_folder_yolo(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select YOLO directory')
        self.txt_yolo.setText(folder)

    def darknet2GUI(self, output):
        self.plainTextEdit.appendPlainText(output.strip()) 
        if self.num_update == 100:
            self.num_update = 0
            #print("Chart updated") chart_network
            if os.path.exists(self.yp+os.path.sep+"chart_network.png"):
                self.train_image_gui.Image.setPixmap(QPixmap(self.yp+os.path.sep+"chart_network.png").scaled(self.train_image_gui.size()));
        self.num_update = self.num_update + 1

    def auxiliar(self, update):
        self.send_message = False

    def command_finnished(self, output):
        if self.send_message:
            self.btn_start.click()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Please verify the chart evolution. You will have it in the Network path.")
            msg.setWindowTitle("Network trained!")
            returnValue = msg.exec()
            if os.path.exists(self.yp+os.path.sep+"chart_network.png"):
                shutil.copyfile(self.yp+os.path.sep+"chart_network.png", self.yp+os.path.sep+"Data"+os.path.sep+self.dp+os.path.sep+"Evolution_chart.png")

    def train(self):
        global process
        if self.btn_start.text() == "TRAIN!":
            self.btn_start.setText("STOP")
            self.plainTextEdit.setPlainText("")
            self.plainTextEdit.setEnabled(True)
            self.train_image_gui.show()

            dataname = self.txt_dataset.text()[self.txt_dataset.text().rfind(os.path.sep)+1:]
            if dataname.endswith(os.path.sep):
                dataname = dataname[:-1]        
            self.dp = dataname[dataname.rfind(os.path.sep)+1:]
            self.num_update = 0
            yolopath = self.txt_yolo.text()
            if yolopath.endswith(os.path.sep):
                yolopath = yolopath[:-1]        
            self.yp = yolopath

            ip = "detector -map train "+yolopath+os.path.sep+"Data"+os.path.sep+str(dataname)+os.path.sep+str(dataname)+".data "+yolopath+os.path.sep+"Data"+os.path.sep+str(dataname)+os.path.sep+"network.cfg "+yolopath+os.path.sep+"Data"+os.path.sep+str(dataname)+os.path.sep+"weights"+os.path.sep+"pre.weight -clear 1"
            print(ip)
            self.send_message = True
            command = yolopath+os.path.sep+'darknet '+ip
            self.worker.run_command(command, cwd="./")

        else:
            self.btn_start.setText("TRAIN!")
            self.plainTextEdit.setEnabled(False)
            self.train_image_gui.hide()
            self.worker.kill(self.yp, self.dp)

    def closeEvent(self, event):
        self.train_image_gui.hide()
        self.closed.emit()

class Worker(QtCore.QObject):
    outSignal = QtCore.pyqtSignal(str)
    outEnd = QtCore.pyqtSignal(str)
    outAux = QtCore.pyqtSignal(int)

    def run_command(self, cmd, **kwargs):
        self.thread = threading.Thread(
            target=self._execute_command, args=(cmd,), kwargs=kwargs, daemon=True
        ).start()

    def _execute_command(self, cmd, **kwargs):
        self.process = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
        )
        for line in self.process.stdout:
            self.outSignal.emit(line.decode())
        self.outEnd.emit("End")

    def kill(self, yp, dp):
        self.outAux.emit(1)
        self.process.terminate()
        if os.path.exists(yp+os.path.sep+"chart_network.png"):
            shutil.copyfile(yp+os.path.sep+"chart_network.png", yp+os.path.sep+"Data"+os.path.sep+dp+os.path.sep+"Evolution_chart.png")

        thread_id = self.thread.ident
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

class Worker2(QObject):
    outSignal = pyqtSignal(str)
    errSignal = pyqtSignal(str)

    def run_command(self, cmd, path):
        proc = QProcess(self)
        proc.setWorkingDirectory(path)
        proc.readyReadStandardOutput.connect(self.onReadyStandardOutput)
        proc.readyReadStandardError.connect(self.onReadyStandardError)
        proc.finished.connect(proc.deleteLater)
        proc.start(cmd)

    @pyqtSlot()
    def onReadyStandardOutput(self):
        proc = self.sender()
        result = proc.readAllStandardOutput().data().decode()
        self.outSignal.emit(result)

    @pyqtSlot()
    def onReadyStandardError(self):
        proc = self.sender()
        result = proc.readAllStandardError().data().decode()
        self.errSignal.emit(result)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GUI_train()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
