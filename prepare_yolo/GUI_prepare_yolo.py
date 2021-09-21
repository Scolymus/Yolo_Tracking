# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

# Form implementation generated from reading ui file 'prepare_yolo.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import prepare_yolo.main as start_yolo
import urllib.request	#install urllib3!
import shutil
import os, sys

class GUI_prepare_yolo(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        uic.loadUi(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"GUI"+os.path.sep+'prepare_yolo.ui', self)

        # Actions changed from designer
        self.btn_browse_dataset.clicked.connect(self.browse_folder_data)
        self.btn_browse_yolo.clicked.connect(self.browse_folder_yolo)
        self.btn_start.clicked.connect(self.setup_yolo_start)
        self.rdb_v4t.toggled.connect(self.change_scheme)
        self.rdb_v3t.toggled.connect(self.change_scheme)
        self.rdb_v3.toggled.connect(self.change_scheme)
        self.rdb_v4.toggled.connect(self.change_scheme)
        self.chk_tiny.setEnabled(False)
        self.spbd_momentum.setProperty("value", 0.9)
        self.spbd_decay.setProperty("value", 0.0005)
        self.spbd_learning.setProperty("value", 0.00261)
        self.spb_burn.setProperty("value", 1000)
        self.spb_w.setProperty("value", 416)
        self.spb_h.setProperty("value", 416)
    
    def browse_folder_data(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select dataset directory')
        self.txt_dataset.setText(folder)

    def browse_folder_yolo(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select YOLO directory')
        self.txt_yolo.setText(folder)

    def setup_yolo_start(self):
        self.btn_start.setEnabled(False)

        file_to_download = ""

        if self.rdb_v4t.isChecked():
            if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4-tiny.weights"):	
                file_to_download = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
                download = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4-tiny.weights"
        elif self.rdb_v3t.isChecked():
            if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov3-tiny.weights"):	
                file_to_download = "https://pjreddie.com/media/files/yolov3-tiny.weights"
                download = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov3-tiny.weights"
        elif self.rdb_v4.isChecked():
            if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4.weights"):	
                file_to_download = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
                download = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4.weights"
        elif self.rdb_v3.isChecked():
            if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov3.weights"):	
                file_to_download = "https://pjreddie.com/media/files/yolov3.weights"
                download = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov3.weights"

        if file_to_download != "":
            print("Downloading required files... This may request some time because they can be as huge as 300 MB.")

            with urllib.request.urlopen(file_to_download) as response, open(download, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            print("Downloaded!!!")

        start_yolo.start(self)
        self.btn_start.setEnabled(True)

    def change_scheme(self):
        if self.rdb_v4.isChecked() or self.rdb_v3.isChecked():
            self.chk_tiny.setEnabled(True)
        else:
            self.chk_tiny.setEnabled(False)

        if self.rdb_v4t.isChecked():
            self.spbd_momentum.setProperty("value", 0.9)
            self.spbd_decay.setProperty("value", 0.0005)
            self.spbd_learning.setProperty("value", 0.00261)
            self.spb_burn.setProperty("value", 1000)
            self.spb_w.setProperty("value", 416)
            self.spb_h.setProperty("value", 416)
        elif self.rdb_v3t.isChecked():
            self.spbd_momentum.setProperty("value", 0.9)
            self.spbd_decay.setProperty("value", 0.0005)
            self.spbd_learning.setProperty("value", 0.001)
            self.spb_burn.setProperty("value", 1000)
            self.spb_w.setProperty("value", 416)
            self.spb_h.setProperty("value", 416)
        elif self.rdb_v4.isChecked():
            self.spbd_momentum.setProperty("value", 0.949)
            self.spbd_decay.setProperty("value", 0.0005)
            self.spbd_learning.setProperty("value", 0.0013)
            self.spb_burn.setProperty("value", 1000)
            self.spb_w.setProperty("value", 608)
            self.spb_h.setProperty("value", 608)
        elif self.rdb_v3.isChecked():
            self.spbd_momentum.setProperty("value", 0.9)
            self.spbd_decay.setProperty("value", 0.0005)
            self.spbd_learning.setProperty("value", 0.001)
            self.spb_burn.setProperty("value", 1000)
            self.spb_w.setProperty("value", 416)
            self.spb_h.setProperty("value", 416)

    def closeEvent(self, event):
        self.closed.emit()
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GUI_prepare_yolo()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
