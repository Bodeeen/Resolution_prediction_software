import sys

from PyQt5.QtWidgets import QApplication

import frcpredict
from frcpredict.ui import MainWindow
from frcpredict.ui.util import initUserFilesIfNeeded


# Init user files if necessary
initUserFilesIfNeeded()

# Show main window
application = QApplication([])
main_window = MainWindow(application.desktop().screenGeometry())
main_window.setWindowTitle(frcpredict.__summary__)
main_window.show()
sys.exit(application.exec_())
