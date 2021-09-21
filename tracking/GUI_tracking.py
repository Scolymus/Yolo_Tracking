# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

# Form implementation generated from reading ui file 'tracking.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import os, sys
import tracking.main as start_tracking

class GUI_tracking(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        uic.loadUi(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"GUI"+os.path.sep+'tracking.ui', self)

        self.btn_browse_dataset.clicked.connect(self.browse_folder_data)
        self.btn_browse_videos.clicked.connect(self.browse_folder_videos)
        self.btn_start.clicked.connect(self.tracking)
        self.chk_run.toggled.connect(lambda: self.change_method(1))
        self.chk_load.toggled.connect(lambda: self.change_method(2))

    def browse_folder_data(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select dataset directory')
        self.txt_dataset.setText(folder)

    def browse_folder_videos(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, 'Select videos directory')
        self.txt_videos.setText(folder)

    def tracking(self):
        type_of_tracking = 0
        if self.chk_run.isChecked():
            type_of_tracking = 1
        elif self.chk_load.isChecked():
            type_of_tracking = 2

        dataname = self.txt_dataset.text()
        if dataname.endswith(os.path.sep):
            dataname = dataname[:-1]       

        videos = self.txt_videos.text()
        if videos.endswith(os.path.sep):
            videos = videos[:-1] 

        print(dataname)
        print(videos)

        start_tracking.init_tracking(self.spb_particles.value(), int(self.spb_pw.value()/2), int(self.spb_ph.value()/2), self.spb_w.value(), self.spb_h.value(), 
                                     self.spb_wf.value(), self.spb_hf.value(), int(self.spb_ww.value()/2), int(self.spb_hw.value()/2), 
                                     type_of_tracking, self.chk_CUDA.isChecked(), dataname, videos)

    def change_method(self, chk):
        if chk == 1 and self.chk_run.isChecked():
            self.chk_load.setChecked(False)
        elif chk == 2 and self.chk_load.isChecked():
            self.chk_run.setChecked(False)

    def closeEvent(self, event):
        self.closed.emit()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GUI_tracking()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

