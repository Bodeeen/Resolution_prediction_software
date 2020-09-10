# Resolution Prediction Software

## Requirements
 - Python 3.7 or later
 - The packages specified by [requirements.txt](requirements.txt). Running the command
   `pip install -r requirements.txt` in the root folder of the repo will install these. If you have
   any issues with pip failing to build the packages on Windows, you may find prebuilt versions
   [here](https://www.lfd.uci.edu/~gohlke/pythonlibs).
   
**If you are using Windows and have an Intel processor**, it is recommended to install the numpy
build that's available [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) (pick the latest
version, with "cp" corresponding to your Python version, and "win32"/"win_amd64" depending on
whether you have a 32-bit or 64-bit processor, and then install it using `pip install`). This is
because of certain issues that may arise with the numpy build that's available on PyPI (and
automatically downloaded when you run the command above) even if it installs correctly. 

## Running the application
In the root folder of the repo, run the command `python -m frcpredict` to launch the software.

## Further information
See [the wiki](https://github.com/Bodeeen/Resolution_prediction_software/wiki) for more information
about how to use the program as well as a developer guide.
 