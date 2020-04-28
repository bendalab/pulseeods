import sys
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import load_data
from thunderfish.bestwindow import best_window
import pulse_tracker_helper as pth

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

def extract_eod_times(data,thresh,peakwidth):
    
    print('extracting times')

    pk, tr = ed.detect_peaks(data, thresh)

    if len(pk)==0:
        return []
    else:
        peaks = pth.makeeventlist(pk,tr,data,peakwidth)
        peakindices, _, _ = pth.discardnearbyevents(peaks[0],peaks[1],peakwidth)
        return peakindices

def extract_pulsefish(data, samplerate):
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
    # 1. extract peaks
    idx_arr, elec_masks = extract_eod_times(data,peak_detection_threshold,peakwidth/dt)
    # 2. cluster the extracted eods.
    
    return [], []


# pulse extraction:
mean_eods, eod_times = extract_pulsefish(data, samplerate)