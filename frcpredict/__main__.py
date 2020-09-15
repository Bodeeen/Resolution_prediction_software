import sys

from PyQt5.QtWidgets import QApplication

from frcpredict.ui import MainWindow
from frcpredict.ui.util import initUserFilesIfNeeded


# Init user files if necessary
initUserFilesIfNeeded()

# Show main window
application = QApplication([])
mainWindow = MainWindow(application.desktop().screenGeometry())
mainWindow.show()
sys.exit(application.exec_())
