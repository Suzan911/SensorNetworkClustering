# 
# Generate an array object with random data.
# Bandpass filter, compute spectrograms, block average, and plot the result.
# 
# 
# 


import numpy as np
import ArrayData as ad
import spectrogram as sg
import windows
import nriplotting as npl

from numpy.lib.stride_tricks import as_strided


# FFT parameters
nFFT = 256
nHop = 256

# Number of windows
M = 5000

# Number of signal sources
nSignals = 10

# Coordinates of sources
coords = np.random.randn(nSignals,2)

# Synthesize random, independent signals
D = np.random.randn(nHop*M,nSignals)
dt = 0.0041  # Define sampling period
fRange = [10,80]  # Define Frequency range of interest

# Create an array data object
sigs = ad.ArraySig(D,dt,coords=coords)

# Bandpass filter the traces in the array object
band = [20,30]
sigs.filterData(band)


# Create an array spectrogram object from time-series object
arraySpectra = ad.ArraySpectrogram(timeseries=sigs,nFFT=nFFT,nHop=nHop,windowFn=windows.hann,sampPeriod=dt,freqRange=fRange)

# Perfom block averaging on spectrograms
arraySpectra.blockAverage(7,3)


# Plot one of the spectrograms
# 
from matplotlib import pyplot as plt

# Get a particular spectrogram
spgrm = arraySpectra.getSpectra(8)

# Extract spectrogram data
S = spgrm.getSpectra()
# Get frequency vector
f = spgrm.getFreqVec()

# 
# S = np.real(S*np.conj(S))

S = arraySpectra.getCohMatrix(20,7,6)
S = np.real(S*np.conj(S))



# Create plotting object
myfig = npl.myfig(
                  subplotw=10,
                  subploth=[10],
                  yspace=1.,
                  x0=1,y0=.8,
                  x1=.25,y1=.8)

# Create axes
ax = myfig.makeAxes(0)
plt.sca(ax)

mark_size = 8

font_title = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'bold'}
font_label = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal'}

plt.imshow(S.T, interpolation="nearest", aspect='auto', cmap=plt.get_cmap("Blues"))


# plt.legend(['Calib true','Float true','Source true','Calib est','Float est','Source est'],
#             fontsize=8,loc='best')
#
# ll = np.min(X,axis=0)+[-15.,-15.]
# ur = np.max(X,axis=0)+[ 15., 15.]
# plt.xlim((ll[0],ur[0]))
# plt.ylim((ll[1],ur[1]))

plt.title('Array coherence matrix',**font_title)
plt.xlabel('Sensor ID',**font_label)
plt.ylabel('Sensor ID',**font_label)

# Output figure as PDF
fnameout = 'out.pdf'
plt.savefig(fnameout)



