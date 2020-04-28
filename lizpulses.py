import sys
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import load_data
from thunderfish.bestwindow import best_window
import pulse_tracker_helper as pth
import thunderfish.eventdetection as ed

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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
    
    thresh = np.mean(np.abs(data))*2
    print(thresh)
    pk, tr = ed.detect_peaks(data, thresh)

    if len(pk)==0:
        return [], []
    else:
        peaks = pth.makeeventlist(pk,tr,data,peakwidth)
        peakindices, _, _ = pth.discardnearbyevents(peaks[0],peaks[1],peakwidth)
        return peaks[0][peakindices.astype('int')]

def extract_pulsefish(data, samplerate, thresh=0.01):
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
    
    ms, vs, ts = [], [], []
    
    
    # this would be the maximum pulsewidth. (used for peak extraction) in seconds
    pw=0.002
    
    # this is the cutwidth (used for snippet extraction) in seconds
    cw=0.001
    
    # 1. extract peaks
    idx_arr = extract_eod_times(data, thresh, pw*samplerate)
    
        
    if len(idx_arr) > 0:
    
        # 2. extract snippets
        idx_arr = idx_arr[(idx_arr>int(cw*samplerate/2)) & (idx_arr<(len(data)-int(cw*samplerate/2)))]
        snippets = np.stack([data[int(idx-cw*samplerate/2):int(idx+cw*samplerate/2)] for idx in idx_arr])

        # 3. pre-process snippets
        snippets = normalize(snippets)


        # 4. extract relevant snippet features
        pca = PCA(10).fit(snippets).transform(snippets)

        # 5. cluster snippets based on waveform
        # EODs are now only clustered based on normalized waveform
        # amplitudes are neglected.

        c = DBSCAN(eps=0.2,min_samples=10).fit(pca).labels_

        # 6. for each cluster, extract the most meaningful time window
        # try time windows from 10 to 100 samples removed from the mean.
        # I could adapt this to scale with samplerate.

        lw = 10

        for l in np.unique(c):
            if l != -1:  

                rs = []

                for rw in range(10,100):

                    # try different windows and different time shifts.
                    # use only indexes that fit with the cutwidth

                    c_i = idx_arr[c== l][(idx_arr[c== l]>lw) & (idx_arr[c== l]<(len(data)-rw))]

                    w = np.stack([data[int(idx-lw):int(idx+rw)] for idx in c_i])

                    m = np.mean(w,axis=0)
                    v = np.std(w,axis=0)
                    r = np.var(m)/np.mean(v)

                    rs.append(r)

                rw = (np.argmax(rs) + 10)

                rs = []

                for lw in range(10,100):
                    # try different windows and different time shifts.
                    c_i = idx_arr[c== l][(idx_arr[c== l]>lw) & (idx_arr[c== l]<(len(data)-rw))]
                    w = np.stack([data[int(idx-lw):int(idx+rw)] for idx in c_i])

                    m = np.mean(w,axis=0)
                    v = np.std(w,axis=0)
                    r = np.var(m)/np.mean(v)

                    rs.append(r)

                lw = (np.argmax(rs) + 10)


                # if the error is small enough, it is probably not noise
                if np.max(rs) > 0.005:
                    c_i = idx_arr[c== l][(idx_arr[c== l]>lw*4) & (idx_arr[c== l]<(len(data)-rw*3))]

                    w = np.stack([data[int(idx-lw*4):int(idx+rw*3)] for idx in c_i])
                    ms.append(np.mean(w,axis=0))
                    vs.append(np.std(w,axis=0))

                    ts.append(idx_arr[c==l])
    
    return ms, vs, ts

def plot_timepoints(data,eod_times,fs):
    plt.figure()
    plt.plot(np.arange(len(data))/fs,data)
    for i,t in enumerate(eod_times):
        plt.plot(t/fs,data[t.astype('int')],'o',label=i+1)
    plt.xlabel('time [s]')
    plt.ylabel('amplitude [V]')
    plt.title('detected EODs')
    plt.legend()
    plt.show()

def plot_eods(mean_eods,eod_std,fs):
    for i, (m,v) in enumerate(zip(mean_eods,eod_std)):
        plt.figure()
        plt.plot(1000*np.arange(len(m))/fs, m,c='k')
        plt.fill_between(1000*np.arange(len(m))/fs,m-v,m+v)
        plt.xlabel('time [ms]')
        plt.ylabel('amplitude [V]')
        plt.title('EOD #%i'%(i+1))
        plt.show()

# pulse extraction:
mean_eods, eod_std, eod_times = extract_pulsefish(data, samplerate)

plot_timepoints(data,eod_times,samplerate)
plot_eods(mean_eods,eod_std,samplerate)