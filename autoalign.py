#! /usr/bin/python
from __future__ import print_function, division
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys

class MeterSignal(object):
    def __init__(self, filename):
        """Loads REDD-formatted data. Returns pandas Timeseries."""
        print("Loading", filename, "...", end="")
        self.data = np.loadtxt(open(filename, 'rU'), dtype=[('timestamp', np.uint32), ('watts', float)])
        self.f_diff = self.data['watts'][1:] - self.data['watts'][:-1]
        self.delta_t = self.data['timestamp'][1:] - self.data['timestamp'][:-1]
        print("done.")

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
            end_i = self.index_from_timestamp(end_time)
            
            # get the indices of the 10 largest forward diffs in this window 
            top_f_diffs = np.argsort(self.f_diff[start_i:end_i])[-10:]
            
            se_acc = 0 # squared error accumulator
            while self.data['timestamp'][i+1] < end_time:
                haystack_f_diff  = self.f_diff[i]
                haystack_delta_t = self.delta_t[i]
                
                target_start_i = self.data['timestamp'][i] - self.data['timestamp'][start_i]
                target_end_i = target_start_i + haystack_delta_t
                
                target_f_diff = watts_up.data[target_end_i] - watts_up.data[target_start_i]
                 
                se_acc += (target_f_diff - haystack_f_diff)**2
                i += 1
                
            mse = se_acc / (i - start_i) # mean squared error
            if mse < best['mse']:
                best['mse'] = mse
                best['start_index'] = start_i
                print(best)
                
            print("{:.2%}".format(start_i / self.data.size), end="\r")
            sys.stdout.flush()
        
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
        self._trim()
        self.steady_states = self._calc_steady_states()
        self.steady_state_transitions = self._calc_good_steady_state_transitions()

    def plot(self, axes, start_timestamp=0):
        x = [datetime.datetime.fromtimestamp(t+start_timestamp) for t in range(0, self.data.size, self.period)]
        axes.plot(x, self.data)
        
    def plot_steady_states(self, axes):
        for start, end, value in self.steady_states:
            line = matplotlib.lines.Line2D([start, end], [value, value], color='r')
            axes.add_line(line)

    @property
    def duration(self):
        """Return duration in seconds."""
        return self.data.size * self.period
    
    def _trim(self):
        """Top and tail data to remove values <= 2 watts."""
        high = self.data > 2

        for trim_front in range(0, self.data.size):
            if high[trim_front]:
                break
            
        for trim_back in range(self.data.size-1, trim_front, -1):
            if high[trim_back]:
                break

        self.data = self.data[trim_front:trim_back]
    
    def _calc_steady_states(self):
        """Return a list of 3-element tuples: (start index, end index, value)"""       
        TOL = 10 # tollerance in Watts
        MIN_LEN = 12 # min length of a steady state
        av = self.data[0]
        acc = self.data[0] # accumulator (for average)
        start_i = 0
        states = []
        for i in range(1, self.data.size):
            if av-TOL < self.data[i] < av+TOL:
                acc += self.data[i]
                av = acc / (1 + i - start_i)
            else:
                if (i-start_i) > MIN_LEN:
                    states.append((start_i, i, av))
                acc = self.data[i]
                av = self.data[i]
                start_i = i

        return states
    
    def _calc_good_steady_state_transitions(self):
        """Good state transitions are those which are large in magnitude
        and where the two states are very close"""
        ss_transitions = [] # list of tuples for transition: start, length, value, score
        last_ss_end = 0
        last_ss_value = 0
        for ss_start, ss_end, ss_value in self.steady_states:
            t_start = last_ss_end
            t_length = ss_start - last_ss_end
            if t_length == 0:
                t_length = 1
            t_value = ss_value - last_ss_value
            t_score = abs(t_value) if t_length < 18 else abs(t_value)/t_length
            
            ss_transitions.append((t_start, t_length, t_value, t_score))
            last_ss_end = ss_end
            last_ss_value = ss_value
            
        return sorted(ss_transitions, key=lambda score: score[3], reverse=True)


if __name__ == "__main__":
    haystack = MeterSignal("/home/jack/Dropbox/Data/data0/channel_01.dat")
    target = WattsUp("/home/jack/Dropbox/Data/BellendenWMNov2012.TXT")
    
    print(target.steady_states)
    print(target.steady_state_transitions)
    
    
    #best = haystack.locate_signal(target)
    #print(best)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(target.data)
    target.plot_steady_states(ax)
    
    #haystack.plot(ax)
    #target.plot(ax, haystack.data['timestamp'][25387])
    plt.show()
    