# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

# Form implementation generated from reading ui file 'dataset.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import create_dataset.main as start_functions
import os, sys

class GUI_dataset(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        global working_images

        uic.loadUi(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"GUI"+os.path.sep+'dataset.ui', self)

        # Values changed from designer
        self.spn_classes.setMinimum(1)
        self.spn_size_w.setMinimum(1)
        self.spn_size_h.setMinimum(1)
        self.spn_frames_per_video.setMinimum(1)
        self.prb_images.setProperty("value", 0)
        self.prb_images.setMaximum(100)

        # Actions changed from designer
        self.btn_browse_in.clicked.connect(self.browse_folder_in)
        self.btn_browse_out.clicked.connect(self.browse_folder_out)
        self.btn_start.clicked.connect(self.create_dataset_start)
        self.spn_size_w.valueChanged.connect(self.changing_size_fixed_ROI)
        self.spn_size_h.valueChanged.connect(self.changing_size_fixed_ROI)
        self.spn_frames_per_video.valueChanged.connect(self.changing_size_fixed_ROI)
        working_images = False

    def browse_folder_in(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select videos directory')
        self.txt_in.setText(folder)
        if self.txt_out.text() == "":
            self.txt_out.setText(folder)

    def browse_folder_out(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select videos directory')
        self.txt_out.setText(folder)

    def create_dataset_start(self):
        global working_images
        self.btn_start.setEnabled(False)
        working_images = True
        working_images = start_functions.start(self, self.txt_in.text(), self.txt_out.text(), self.chk_read.isChecked(), self.spn_classes.value(), self.spn_size_w.value(), self.spn_size_h.value(), self.spn_frames_per_video.value())

    def changing_size_fixed_ROI(self):
        if working_images:
            start_functions.change_size_fixed_ROI(self.spn_size_w.value(), self.spn_size_h.value(), self.spn_frames_per_video.value())

    def closeEvent(self, event):
        self.closed.emit()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GUI_dataset()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

