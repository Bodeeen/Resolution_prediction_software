import sys
from PyQt5.QtWidgets import QApplication
import frcpredict
from frcpredict.ui import MainWindow

app = QApplication([])
main_window = MainWindow()
main_window.setWindowTitle(frcpredict.__summary__)
main_window.show()
sys.exit(app.exec_())
