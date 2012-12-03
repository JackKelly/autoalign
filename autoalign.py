from __future__ import print_function, division
import pandas
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Load timestamped data (haystack)
# Load WattsUp data (target)
# for i in range(0, haystack.end - target.duration):
#    for j in range(i, :

# Keep a best score and the location for that best score

def load_haystack(filename):
    """Loads REDD-formatted data. Returns pandas Timeseries."""
    print("Loading", filename)
    
    dateparser = lambda timestamp: datetime.datetime.fromtimestamp(int(timestamp))
    return pandas.read_csv(filename, sep=" ", index_col=0, parse_dates=True, date_parser=dateparser, squeeze=True, header=None)


def load_target(filename):
    """Load WattsUp data"""
    wu = pandas.read_csv(filename, sep="\t", skiprows=2, 
                         names=["Time", "Watts", "Volts", "Amps", "WattHrs", "Cost", 
                                "Avg Kwh", "Mo Cost", "Max Wts", "Max Vlt", "Max Amp",
                                "Min Wts", "Min Vlt", "Min Amp", "Pwr Fct", "Dty Cyc",
                                "Pwr Cyc", "A", "B"])
    
    wu = wu['Watts']
    
    index = pandas.Index([datetime.timedelta(seconds=int(secs)) for secs in wu.index])
    
    return pandas.Series(wu.values, index=index)


def plot_target(target, start_date, axes):
    # x = np.array([start_date + datetime.timedelta(seconds=secs) for secs in target.index])
    axes.plot(target.index + start_date, target.values)


if __name__ == "__main__":
    s = load_haystack("/homes/dk3810/Dropbox/Data/data0/channel_01.dat")
    wu = load_target("/homes/dk3810/Dropbox/Data/BellendenWMNov2012.TXT")
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(s.index, s.values)
    
    plot_target(wu, datetime.datetime(year=2012, month=11, day=10), ax)

    plt.show()
    