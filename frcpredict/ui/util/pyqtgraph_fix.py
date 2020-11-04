import struct

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtVersion
from pyqtgraph.functions import eq


def fixPyQtGraphNanBehaviour() -> None:
    """
    Overrides the arrayToQPath function in pyqtgraph to allow plotting np.nan values. The code comes
    from the following pull request: https://github.com/pyqtgraph/pyqtgraph/pull/1287
    """

    global _hasOverriddenPyQtGraphArrayToQPath
    if _hasOverriddenPyQtGraphArrayToQPath:
        return

    def arrayToQPath(x, y, connect='all'):
        """Convert an array of x,y coordinats to QPainterPath as efficiently as possible.
        The *connect* argument may be 'all', indicating that each point should be
        connected to the next; 'pairs', indicating that each pair of points
        should be connected, or an array of int32 values (0 or 1) indicating
        connections.
        """

        ## Create all vertices in path. The method used below creates a binary format so that all
        ## vertices can be read in at once. This binary format may change in future versions of Qt,
        ## so the original (slower) method is left here for emergencies:
        # path.moveTo(x[0], y[0])
        # if connect == 'all':
        # for i in range(1, y.shape[0]):
        # path.lineTo(x[i], y[i])
        # elif connect == 'pairs':
        # for i in range(1, y.shape[0]):
        # if i%2 == 0:
        # path.lineTo(x[i], y[i])
        # else:
        # path.moveTo(x[i], y[i])
        # elif isinstance(connect, np.ndarray):
        # for i in range(1, y.shape[0]):
        # if connect[i] == 1:
        # path.lineTo(x[i], y[i])
        # else:
        # path.moveTo(x[i], y[i])
        # else:
        # raise Exception('connect argument must be "all", "pairs", or array')

        ## Speed this up using >> operator
        ## Format is:
        ##    numVerts(i4)
        ##    0(i4)   x(f8)   y(f8)    <-- 0 means this vertex does not connect
        ##    1(i4)   x(f8)   y(f8)    <-- 1 means this vertex connects to the previous vertex
        ##    ...
        ##    cStart(i4)   fillRule(i4)
        ##
        ## see: https://github.com/qt/qtbase/blob/dev/src/gui/painting/qpainterpath.cpp

        ## All values are big endian--pack using struct.pack('>d') or struct.pack('>i')

        path = QtGui.QPainterPath()

        n = x.shape[0]

        # create empty array, pad with extra space on either end
        arr = np.empty(n + 2, dtype=[('c', '>i4'), ('x', '>f8'), ('y', '>f8')])

        # write first two integers
        byteview = arr.view(dtype=np.ubyte)
        byteview[:16] = 0
        byteview.data[16:20] = struct.pack('>i', n)

        # Fill array with vertex values
        arr[1:-1]['x'] = x
        arr[1:-1]['y'] = y

        # inf/nans completely prevent the plot from being displayed starting on
        # Qt version 5.12.3; these must now be manually cleaned out.
        isfinite = None
        qtver = [int(x) for x in QtVersion.split('.')]
        if qtver >= [5, 12, 3]:
            isfinite = np.isfinite(x) & np.isfinite(y)
            if not np.all(isfinite):
                # credit: Divakar https://stackoverflow.com/a/41191127/643629
                mask = ~isfinite
                idx = np.arange(len(x))
                idx[mask] = -1
                np.maximum.accumulate(idx, out=idx)
                first = np.searchsorted(idx, 0)
                if first < len(x):
                    # Replace all non-finite entries from beginning of arr with the first finite one
                    idx[:first] = first
                    arr[1:-1] = arr[1:-1][idx]

        # decide which points are connected by lines
        if eq(connect, 'all'):
            arr[1:-1]['c'] = 1
        elif eq(connect, 'pairs'):
            arr[1:-1]['c'][::2] = 0
            arr[1:-1]['c'][1::2] = 1  # connect every 2nd point to every 1st one
        elif eq(connect, 'finite'):
            # Let's call a point with either x or y being nan is an invalid point.
            # A point will anyway not connect to an invalid point regardless of the
            # 'c' value of the invalid point. Therefore, we should set 'c' to 0 for
            # the next point of an invalid point.
            if isfinite is None:
                isfinite = np.isfinite(x) & np.isfinite(y)
            arr[2:]['c'] = isfinite
        elif isinstance(connect, np.ndarray):
            arr[1:-1]['c'] = connect
        else:
            raise Exception('connect argument must be "all", "pairs", "finite", or array')

        arr[1]['c'] = 0  # the first vertex has no previous vertex to connect

        byteview.data[-20:-16] = struct.pack('>i', 0)  # cStart
        byteview.data[-16:-12] = struct.pack('>i', 0)  # fillRule (Qt.OddEvenFill)

        # create datastream object and stream into path

        ## Avoiding this method because QByteArray(str) leaks memory in PySide
        # buf = QtCore.QByteArray(arr.data[12:lastInd+4])  # I think one unnecessary copy happens here

        path.strn = byteview.data[16:-12]  # make sure data doesn't run away
        try:
            buf = QtCore.QByteArray.fromRawData(path.strn)
        except TypeError:
            buf = QtCore.QByteArray(bytes(path.strn))

        ds = QtCore.QDataStream(buf)
        ds >> path

        return path

    pg.functions.arrayToQPath = arrayToQPath
    _hasOverriddenPyQtGraphArrayToQPath = True


_hasOverriddenPyQtGraphArrayToQPath = False
