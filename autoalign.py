#! /usr/bin/python
from __future__ import print_function, division
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys

"""
Attempts to automatically align a WattsUp power signature to a Current Cost
power time series.

This script is just a bit of tinkering.  The algorithm is far too basic
and fragile to be of much use to anyone.  Find steady states, then
steady state transitions, and then try to match these.  It would probably 
be easier and more reliable to just look at large-magnitude forward diffs.
"""

def calc_steady_states(data, timestamps=None):
    """Return a list of 3-element tuples: (start index, end index, value)"""       
    TOL = 10 # tolerance in watts
    MIN_LEN = 12 # min length of a steady state
    
    av = data[0] # average (mean)
    acc = data[0] # accumulator (for average)
    start_i = 0
    states = []
    for i in range(1, data.size):
        if av-TOL < data[i] < av+TOL:
            acc += data[i]
            av = acc / (1 + i - start_i)
        else:            
            duration = i-start_i if timestamps is None else \
                       timestamps[i]-timestamps[start_i]

            if duration > MIN_LEN:
                states.append((start_i, i, av))
            acc = data[i]
            av = data[i]
            start_i = i

    return states


def calc_good_steady_state_transitions(steady_states, timestamps=None):
    """Good state transitions are those which are large in magnitude
    and where the two states are very close.
    
    Returns list of tuples for transition: 
      - start index
      - length (seconds)
      - value (watts)
      - score
    """
      
    MIN_SCORE = 10
    
    ss_transitions = []
    last_ss_end = 0
    last_ss_value = 0
    for ss_start, ss_end, ss_value in steady_states:

        t_length = ss_start - last_ss_end if timestamps is None else \
                   timestamps[ss_start] - timestamps[last_ss_end]
                    
        if t_length == 0:
            t_length = 1
            
        t_value = ss_value - last_ss_value
        t_score = abs(t_value) if t_length < 18 else abs(t_value)/t_length
        last_ss_end = ss_end
        last_ss_value = ss_value
        
        if t_score > MIN_SCORE:
            ss_transitions.append((last_ss_end, t_length, t_value, t_score))
                        
    return ss_transitions


def plot_steady_states(axes, steady_states):
    for start, end, value in steady_states:
        line = matplotlib.lines.Line2D([start, end], [value, value], color='r')
        axes.add_line(line)


class MeterSignal(object):
    def __init__(self, filename):
        """Loads REDD-formatted data. Returns pandas Timeseries."""
        print("Loading", filename, "...", end="")
        self.data = np.loadtxt(open(filename, 'rU'), dtype=[('timestamp', np.uint32), ('watts', float)])
        self.f_diff = self.data['watts'][1:] - self.data['watts'][:-1]
        self.delta_t = self.data['timestamp'][1:] - self.data['timestamp'][:-1]
        self.steady_states = calc_steady_states(self.data['watts'], self.data['timestamp'])
        self.steady_state_transitions = calc_good_steady_state_transitions(self.steady_states, self.data['timestamp'])
        print("done.")

    def plot(self, axes):
        x = [datetime.datetime.fromtimestamp(t) for t in self.data['timestamp']]
        axes.plot(x, self.data['watts']/1000, label="EDF IAM")

    def locate_signal(self, watts_up):
        if watts_up.duration >= self.duration:
            raise ValueError("WattsUp duration {} >= MeterSignal duration {}"
                             .format(watts_up.duration, self.duration))
        
        best = {'mean_error': np.finfo(float).max, # set Mean Sq Error to max float val
                'start_index': 0}
        
        end_time_of_search_window = self.data['timestamp'][-1] - watts_up.duration
        
        for start_i in range(0, len(self.steady_state_transitions)):            
            start_timestamp = self.data['timestamp'][self.steady_state_transitions[start_i][0]]
             
            if start_timestamp > end_time_of_search_window:
                break
            
            error_acc = 0 # error accumulator
            inner_i = start_i
            for target_transition in watts_up.steady_state_transitions:
                # Does a transition exist in haystack t_start_i seconds away from start_timestamp, length t_length, value t_value
                try:
                    closest_match = self.find_transition_matching(target_transition, inner_i, start_timestamp)
                except IndexError:
                    error_acc += 100 # penalise misses
                    continue
                
                error_acc += closest_match['error']
                inner_i = closest_match['i']
                
            mean_error = error_acc / len(watts_up.steady_state_transitions)
            if mean_error and mean_error < best['mean_error']:
                best['mean_error'] = mean_error
                best['start_index'] = start_i
                print(best)
                
            print("{:.2%}".format(start_i / len(self.steady_state_transitions)), end="\r")
            sys.stdout.flush()
        
        return best
    
    def find_transition_matching(self, target_transition, i, start_timestamp):
        best = {'error': np.finfo(float).max, # set Error to max float val
                'i': 0}        
        
        while True:
            candidate_transition_time = self.data['timestamp'][self.steady_state_transitions[i][0]]
            
            target_transition_time = target_transition[0]+start_timestamp
            
            if candidate_transition_time > target_transition_time+20:
                break
            
            error = abs(candidate_transition_time - target_transition_time) + abs(self.steady_state_transitions[i][2] - target_transition[2])   
            
            if error < best['error']:
                best['error'] = error
                best['i'] = i
            
            i += 1
            
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
        self.steady_states = calc_steady_states(self.data)
        self.steady_state_transitions = calc_good_steady_state_transitions(self.steady_states)

    def plot(self, axes, start_timestamp=0):
        x = [datetime.datetime.fromtimestamp(t+start_timestamp) for t in range(0, self.data.size, self.period)]
        axes.plot(x, self.data/1000, color='g', label="WattsUp")
        


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
        av = self.data[0] # average (mean)
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
    
    
def plot_three():
    iam = MeterSignal("/home/jack/Dropbox/Data/data2/channel_5.dat")
    whole_house = MeterSignal("/home/jack/Dropbox/Data/data2/channel_1.dat")
    target = WattsUp("/home/jack/Dropbox/Data/BellendenWMNov2012.TXT")    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    
    
    #ax.plot(target.data)
    #plot_steady_states(ax, target.steady_states)
    
    #ax.plot(haystack.data['watts'])
    #plot_steady_states(ax, haystack.steady_states)    

    iam.plot(ax1)
    whole_house.plot(ax2)
    t=1352822455  # correct start time for WM target is 1352822455 in data2 
    target.plot(ax3,t)
     
    ax3.set_xlabel("time")
    ax2.set_ylabel("power (kw)")
    
    date_formatter = matplotlib.dates.DateFormatter("%H:%M")
    date_formatter_none = matplotlib.dates.DateFormatter("")
    ax1.xaxis.set_major_formatter( date_formatter_none )
    ax2.xaxis.set_major_formatter( date_formatter_none )
    ax3.xaxis.set_major_formatter( date_formatter )
    
    ax1.set_xlim( ax3.get_xlim() )
    ax2.set_xlim( ax3.get_xlim() )
    ax1.set_ylim( ax3.get_ylim() )
    ax2.set_ylim([0, 3])
    
    ax1.set_title("EDF IAM")
    ax2.set_title("EDF whole house")
    ax3.set_title("WattsUp")
    
    plt.show()
        

def auto():
    haystack = MeterSignal("/home/jack/Dropbox/Data/data2/channel_5.dat")
    target = WattsUp("/home/jack/Dropbox/Data/BellendenWMNov2012.TXT")
    
    print(target.steady_states)
    print(target.steady_state_transitions)
    
    best = haystack.locate_signal(target)
    print(best)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    haystack.plot(ax1)
    t= haystack.data['timestamp'][haystack.steady_state_transitions[best['start_index']][0]]
    print("t",t)
    target.plot(ax1,t) 
    ax1.set_xlabel("time")
    ax1.set_ylabel("power (kw)")
    date_formatter = matplotlib.dates.DateFormatter("%d/%m\n%H:%M")
    ax1.xaxis.set_major_formatter( date_formatter )
    ax1.legend()
    plt.show()
    

if __name__ == "__main__":
    plot_three()
    # auto()
    