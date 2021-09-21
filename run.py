# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from create_dataset.GUI_dataset import GUI_dataset
from prepare_yolo.GUI_prepare_yolo import GUI_prepare_yolo
from train_yolo.GUI_train import GUI_train
from tracking.GUI_tracking import GUI_tracking

import os, sys

class GUI_main(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        uic.loadUi(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"GUI"+os.path.sep+'main.ui', self)

        self.btn_create_dataset.clicked.connect(self.open_dataset)
        self.btn_prepare_dataset.clicked.connect(self.open_prepare)
        self.btn_run_train.clicked.connect(self.open_train)        
        self.btn_run_tracking.clicked.connect(self.open_track)

        self.Dataset_yolo = GUI_dataset()
        self.Prepare_yolo = GUI_prepare_yolo()
        self.train_gui = GUI_train()
        self.track_gui = GUI_tracking()

        self.Dataset_yolo.closed.connect(self.show)
        self.Prepare_yolo.closed.connect(self.show)
        self.train_gui.closed.connect(self.show)
        self.track_gui.closed.connect(self.show)

    def open_dataset(self):
        self.Dataset_yolo.show()
        self.hide()

    def open_prepare(self):
        self.Prepare_yolo.show()
        self.hide()

    def open_train(self):        
        self.train_gui.show()
        self.hide()

    def open_track(self):        
        self.track_gui.show()
        self.hide()

    def retranslateUi(self, frame_init):
        _translate = QtCore.QCoreApplication.translate
        frame_init.setWindowTitle(_translate("frame_init", "Track them!"))
        self.btn_create_dataset.setText(_translate("frame_init", "Create Dataset"))
        self.btn_run_train.setText(_translate("frame_init", "Train network"))
        self.btn_prepare_dataset.setText(_translate("frame_init", "Prepare YOLO"))
        self.btn_run_tracking.setText(_translate("frame_init", "Run Tracking"))

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GUI_main()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


