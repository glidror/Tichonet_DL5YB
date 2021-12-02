#**************************************************
# QtPy - GUI example
#**************************************************
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QDialogButtonBox

"""
print ("------------------------------------------------------------------")
print ("Targil 9050 Ex.1 - Quick and Dirty")
print ("------------------------------------------------------------------")
app = QApplication(sys.argv)
win = QMainWindow()
win.setGeometry(150,200,300,40)
win.setWindowTitle("my first PyQt5 window")
label = QtWidgets.QLabel("Hi there",win)
win.show()
sys.exit(app.exec_())

print ("------------------------------------------------------------------")
print ("Targil 9050 Ex.2 - Crfeat an evant handler")
print ("------------------------------------------------------------------")
def button_clicked():
    msg = QMessageBox()
    msg.setWindowTitle("My first popup")
    msg.setText("Hi There")
    msg.exec_()

app = QApplication(sys.argv)
win = QMainWindow()
win.setGeometry(150,200,300,40)
button = QtWidgets.QPushButton("OK",win)
button.clicked.connect(button_clicked)
win.show()
sys.exit(app.exec_())


print ("------------------------------------------------------------------")
print ("Targil 9050 Ex.3")
print ("------------------------------------------------------------------")
class MyMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("PyQt5 class")
        self.setGeometry(150,200,300,100)
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel("Hi there",self)
        self.label.setGeometry(50,20,100,10)
        self.button = QtWidgets.QPushButton("OK",self)
        self.button.setGeometry(10,20,20,10)
        self.button.clicked.connect(self.button_clicked)

    def button_clicked(self):
        msg = QMessageBox()
        msg.setWindowTitle("My first popup")
        msg.setText("popup message")
        msg.exec_()

        print ("------------------------------------------------------------------")
        print ("Targil 9050 Ex.4")
        print ("------------------------------------------------------------------")
        # Pressing the button in the popup window will
        # terminate the popup and set the label in the main window
        self.label.setText("You pressed the button")
        self.label.adjustSize()

app = QApplication(sys.argv)
win = MyMainWindow()
win.show()
sys.exit(app.exec_())

"""

print ("------------------------------------------------------------------")
print ("Targil 9050 Ex.5")
print ("------------------------------------------------------------------")

class MyMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("PyQt5 class")
        self.setGeometry(150,200,300,100)
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel("Hi there",self)
        self.label.setGeometry(110,10,100,20)
        self.button = QtWidgets.QPushButton("OK",self)
        self.button.setGeometry(30,10,60,20)
        self.button.clicked.connect(self.button_clicked)

    def button_clicked(self):
        dlg = MyDialog()
        if dlg.exec():
            self.label.setText("You confirmed the action")
        else:
            self.label.setText("You canceled the action")
        self.label.adjustSize()


class MyDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setWindowTitle("My first dialog!")
        self.setGeometry(180,220,200,100)
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel("Continue?",self)
        self.label.setGeometry(50,20,100,10)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.setGeometry(100,70,100,30)


app = QApplication(sys.argv)
win = MyMainWindow()
win.show()
sys.exit(app.exec_())
