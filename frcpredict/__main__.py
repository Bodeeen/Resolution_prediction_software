import sys

from PyQt5.QtWidgets import QApplication

import frcpredict
from frcpredict.ui import MainWindow
from frcpredict.ui.util import initUserFilesIfNeeded


# Init user files if necessary
initUserFilesIfNeeded()

# Show main window
application = QApplication([])
mainWindow = MainWindow(application.desktop().screenGeometry())
mainWindow.setWindowTitle(frcpredict.__summary__)
mainWindow.show()
sys.exit(application.exec_())
