'''
This module provides a spectrogram class and some useful functions relating to spectrograms.


'''


import numpy as np
import windows
import scipy as sp
from numpy.lib.stride_tricks import as_strided


class Spectrogram:
    '''
    Defines a class for spectrogram.
    Offers utilities to 
    - represent meta-data, 
    - compute spectrograms, 
    - perform block-averaging, 
    - get frequency vector,
    - etc.
    
    
    Spectrogram(timeseries=D,nFFT=nFFT,nHop=nHop,windowFn=windows.hann,dt=dt)
    '''
    
    
    def __init__(self, **kwargs):
        
        # Extract constructor arguments and set defaults
        dt      = kwargs.pop('dt', None)
        nHop    = kwargs.pop('nHop', None)
        nFFT    = kwargs.pop('nFFT', None)
        windowFn = kwargs.pop('windowFn', None)
        t0      = kwargs.pop('t0', None)
        timeseries = kwargs.pop('timeseries', None)
        spectra = kwargs.pop('spectra', None)
        
        
        
        # Initialize empty object
        if len(kwargs)==0:
            self.__S = spectra
            self.dt = dt
            self.nFFT = nFFT
            self.nHop = nHop
            self.windowFn = windowFn
            self.t0 = t0
            return        
        
        # Init: spectrogram from time series
        if timeseries is not None:
            
            if not isinstance(timeseries,np.ndarray):
                print('Timeseries must be numpy array.')
                return
                        
            if hasattr(windowFn, '__call__'):
                # Define window (requires 'windows' module)
                windowFn = windowFn(nFFT)
            
            S = self.computeSpectrogram(timeseries,nFFT,nHop,dt,windowFn)
            
            self.__S = S
            self.dt = dt
            self.t0 = t0
            self.nFFT = nFFT
            self.nHop = nHop
            self.windowFn = windowFn
            self.nHop = nHop
            return
            
            
        # Init: precomputed spectrogram
        if spectra is not None:
            
            self.__S = spectra
            self.dt = dt
            self.nFFT = nFFT
            self.nHop = nHop
            self.windowFn = windowFn
            self.t0 = t0
            
            
                    
    def getSpectra(self):
        return self.__S
        
        
    def setSpectra(self,S):
        self.__S = S
    
    
    def getFreqVec(self):
        return getFreqVec(self.nFFT,1/self.dt)
        
    def getnFFT(self):
        return self.nFFT
    
        
    def blockAvg(self,M,step):
        '''
        Offer instance level block averaging
        '''
        self.__S = blockAverage(self.__S,M,step)    
    
    
    def computeSpectrogram(self,data,nFFT,nHop,dt,windowFn):
        '''
        Compute STFT spectrogram of data in timeseries. Expected numpy.array shape
        (nSamples,nSignals)
        
        Returns: spectra, (nTimes,nFreq,nSignals)
        '''
        
        x = data[:,np.newaxis]
        X = stft(x, nFFT, nHop, transform=np.fft.fft, win=windowFn, zp_back=0, zp_front=0)
        
        return X
        
        

        
def stft(x, L, hop, transform=np.fft.fft, win=None, zp_back=0, zp_front=0):
    '''
    Arguments:
    x: input signal
    L: frame size
    hop: shift size between frames
    transform: the transform routine to apply (default FFT)
    win: the window to apply (default None)
    zp_back: zero padding to apply at the end of the frame
    zp_front: zero padding to apply at the beginning of the frame
    Return:
    The STFT of x
    '''

    # the transform size
    N = L + zp_back + zp_front

    # window needs to be same size as transform
    if (win is not None and len(win) != N):
        print 'Window length need to be equal to frame length + zero padding.'
        sys.exit(-1)

    # reshape
    new_strides = (hop * x.strides[0], x.strides[0])
    new_shape = ((len(x) - L) / hop + 1, L)
    y = as_strided(x, shape=new_shape, strides=new_strides)

    # add the zero-padding
    y = np.concatenate(
        (np.zeros(
            (y.shape[0], zp_front)), y, np.zeros(
            (y.shape[0], zp_back))), axis=1)

    # apply window if needed
    if (win is not None):
        y = win * y
        #y = np.expand_dims(win, 0)*y

    # transform along rows
    Z = transform(y, axis=1)

    # apply transform
    return Z
    

def getFreqVec(N, Fs, centered=False):
    '''
    N: FFT length
    Fs: sampling rate of the signal
    shift: False if the DC is at the beginning, True if the DC is centered
    '''

    # Create a centered vector. The (1-N%2) is to correct for even/odd length
    vec = np.arange(-N / 2 + (1 - N % 2), N / 2 + 1) * float(Fs) / float(N)

    # Shift positive/negative frequencies if needed. Again (1-N%2) for
    # even/odd length
    if centered:
        return vec
    else:
        return np.concatenate((vec[N / 2 - (1 - N % 2):], vec[0:N / 2 - 1]))
        
        
def blockAverage(S,M,step):
    '''
    Block averaging of spectrogram data. 
    M: number of time windows (block size) are averaged along the frequency axis.
    blockHop: number of time windows to advance after each averaging step.
    '''
    
    
    
    # Number of frequencies
    F = S.shape[1]
    
    # Number of stacked time windows
    L = np.floor((S.shape[0]-M)/step)-1
    
    # The data will be copy-stacked, with M-snapshot spectrograms stacked along a 3rd array axis for subsequent averaging along the 1st axis (the snapshot axis)
    # This is attained using the strides-trick: the final shape of the 3D structure is provided and together with a scheme for where to get the data to populate the new 3D structure (for more information see manual for numpy.ndarray.strides)
    new_shape = (M,F,L)
    new_strides = (S.strides[0],S.strides[1],S.strides[0]*step)
    Snew = as_strided(S, shape=new_shape, strides=new_strides)
    
    # Average along the 1st axis (the snapshot axis)
    S = np.mean(Snew,axis=0)
    
    # Reshape to 2nd and 3rd dimensions of 3D array (time and frequency axes)
    return np.reshape(S,new_shape[1:3]).T
            
    