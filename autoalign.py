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

    def locate_signal(self, watts_up):
        if watts_up.duration >= self.duration:
            raise ValueError("WattsUp duration {} >= MeterSignal duration {}"
                             .format(watts_up.duration, self.duration))
        
        best = {'mse': np.finfo(float).max, # set Mean Sq Error to max float val
                'start_index': 0}
        
        last_i = self.index_from_timestamp(self.data['timestamp'][-1] - watts_up.duration)
        
        for start_i in range(0, last_i):
            i = start_i
            end_time = self.data['timestamp'][start_i] + watts_up.duration
            se_acc = 0 # squared error accumulator
            while self.data['timestamp'][i+1] < end_time:
                haystack_f_diff  = self.data['watts'][i+1] - self.data['watts'][i]
                haystack_delta_t = self.data['timestamp'][i+1] - self.data['timestamp'][i]
                
                target_start_i = self.data['timestamp'][i] - self.data['timestamp'][start_i]
                target_end_i = target_start_i + haystack_delta_t
                
                try:
                    target_f_diff = watts_up.data[target_end_i] - watts_up.data[target_start_i]
                except IndexError:
                    print("IndexError.", target_start_i, target_end_i, i)
                    break
                 
                se_acc += (target_f_diff - haystack_f_diff)**2
                i += 1
                
            mse = se_acc / (i - start_i) # mean squared error
            if mse < best['mse']:
                best['mse'] = mse
                best['start_index'] = start_i
                print(best)
        
        return best
    
    @property
    def duration(self):
        return self.data['timestamp'][-1] - self.data['timestamp'][0]
    
    def index_from_timestamp(self, timestamp):
        """Return the index corresponding to timestamp.
        
        If timestamp is outside bounds then ValueError is raised.
        If a perfect match is then the corresponding index is returned.
        Otherwise the nearest match less than timestamp is returned.
        """
        
        if (timestamp > self.data['timestamp'][-1] or
            timestamp < self.data['timestamp'][0]):
            raise ValueError("timestamp {} is out of bounds".format(timestamp))
        
        # guess the index
        i = int(round(((timestamp - self.data['timestamp'][0]) /
                        self.duration) * (self.data.size-1)))

        # search forwards if necessary
        while self.data['timestamp'][i] < timestamp:
            i += 1
        
        # search backwards if necessary
        while self.data['timestamp'][i] > timestamp:
            i -= 1        
                
        return i


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
        """Return duration in seconds."""
        return self.data.size * self.period


if __name__ == "__main__":
    haystack = MeterSignal("/home/jack/Dropbox/Data/data0/channel_01.dat")
    target = WattsUp("/home/jack/Dropbox/Data/BellendenWMNov2012.TXT")
    
    best = haystack.locate_signal(target)
    print(best)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    haystack.plot(ax)
    target.plot(ax, haystack.data['timestamp'][1000])
    plt.show()
    