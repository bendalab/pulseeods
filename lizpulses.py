import sys
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import load_data
from thunderfish.bestwindow import best_window

# load data:
filename = sys.argv[1]
channel = 0
raw_data, samplerate, unit = load_data(filename, channel)

# best_window:
data, clipped = best_window(raw_data, samplerate, win_size=8.0)

# plot the data you should analyze:
time = np.arange(len(data))/samplerate  # in seconds
plt.plot(time, data)
plt.show()


def extract_pulsefish(data, samplerate, **kwargs):
    """
    This is what you should implement! Don't worry about wavefish for now.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
        
    Returns
    -------
    mean_eods: list of 2D arrays
        For each detected fish the average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    """
    return [], []


# pulse extraction:
mean_eods, eod_times = extract_pulsefish(data, samplerate, **kwargs)