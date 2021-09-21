# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

# Form implementation generated from reading ui file 'train_image.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, uic
import os, sys

class GUI_train_image(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        uic.loadUi(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"GUI"+os.path.sep+'train_image.ui', self)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GUI_train_image()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

