# Resolution Prediction Software

## Standalone downloads

The program can be downloaded in standalone format from
[the releases page](https://github.com/Bodeeen/Resolution_prediction_software/releases).
Python and the required packages are included, so that you don't have to follow the "getting
started" guide below. 


## Getting started (non-standalone)

### Requirements
 - Python 3.7 or later
 - The packages specified by [requirements.txt](requirements.txt). Running the command
   `pip install -r requirements.txt` in the root folder of the repo will install these. If you have
   any issues with pip failing to build the packages on Windows, you may find prebuilt versions
   [here](https://www.lfd.uci.edu/~gohlke/pythonlibs).

### Running the application
In the root folder of the repo, run the command `python -m frcpredict` to launch the software.

### Building a bundle
To build a bundle with a runnable executable and dependencies included, run the PyInstaller command:
`pyinstaller frcpredict.spec`. PyInstaller can be downloaded from PyPI using
`pip install pyinstaller`. 


## Further information
See [the wiki](https://github.com/Bodeeen/Resolution_prediction_software/wiki) for more information
about how to use the program as well as a developer guide.
 