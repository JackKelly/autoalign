from __future__ import print_function, division
import datetime
import matplotlib.pyplot as plt
import numpy as np


class MeterSignal(object):
    def __init__(self, filename):
        """Loads REDD-formatted data. Returns pandas Timeseries."""
        print("Loading", filename)
        self.data = np.loadtxt(open(filename, 'rU'), dtype=[('timestamp', np.uint32), ('watts', float)])

    def plot(self, axes):
        x = [datetime.datetime.fromtimestamp(t) for t in self.data['timestamp']]
        axes.plot(x, self.data['watts'])

    def find(self, watts_up):
        best = {'score':0, 'start_index':0}
        
        # last_timestamp_to_search = timestamp corresponding to self.data['timestamp'][-1]-watts_up.duration
        
        for i in range(0, last_timestamp_to_search):
            pass # TODO
        
        return best
        
class WattsUp(object):
    def __init__(self, filename, period=1):
        """Load WattsUp data"""
        print("Loading", filename)
        self.period = period # seconds
        self.data = np.loadtxt(open(filename, 'rU'), delimiter="\t", dtype=float, skiprows=3, usecols=[1])

    def plot(self, axes, start_timestamp=0):
        x = [datetime.datetime.fromtimestamp(t+start_timestamp) for t in range(0, self.data.size, self.period)]
        axes.plot(x, self.data)

    @property
    def duration(self):
        return self.data.size * self.period

if __name__ == "__main__":
    haystack = MeterSignal("/home/jack/Dropbox/Data/data0/channel_01.dat")
    target = WattsUp("/home/jack/Dropbox/Data/BellendenWMNov2012.TXT")
       
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    haystack.plot(ax)
    target.plot(ax, haystack.data['timestamp'][1000])
    plt.show()
    