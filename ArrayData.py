'''
This module provides two classes, one representing array timeseries data (ArraySig), the other array spectral data (ArraySpectrogram).

'''


import numpy as np
import windows
import spectrogram
from scipy import signal as sp


class ArraySig:
    '''
    Contain and operate on a set of signals from an array
    
    '''
    def __init__(self,data,dt,**kwargs):
        
        self.__data = data
        self.__dt = dt
        
        self.__coords = kwargs.get('coords',None)
        self.__ID = kwargs.get('ID',None)
        self.__t0 = kwargs.get('startTime',0)
        
        
    def sampleLength(self):
        return self.__data.shape[0]
        
        
    def numSignals(self):
        return self.__data.shape[1]
        
        
    def sampPeriod(self):
        return self.__dt
        
        
    def getData(self):
        return self.__data
        
    def getCoords(self):
        return self.__coords
        
    def getIDs(self):
        return self.__ID
        
    def filterData(self,band):
        '''
        Bandpass filter the array data
        '''
        
        # Compute sampling rate
        fs = 1/self.__dt
        
        # Compute filter coefficients for given passband
        band = np.array(band)
        b, a = sp.butter(2,band/(fs/2.),btype='bandpass')
        
        # Loop over array signals and filter them
        for i in range(self.__data.shape[1]):
            x = self.__data[:,i,np.newaxis]
            x = sp.filtfilt(b,a,x,axis=0)
            # Store filtered signal back into the array
            self.__data[:,i,np.newaxis] = x[:]
            
            


class ArraySpectrogram:
    '''
    Stores and operates on a set of spectrograms from an array
    '''
    
    # def __init__(self, S,dt,nFFT=None,nHop=none,windowFn=None):
    def __init__(self, **kwargs):
        
        
        self.__nFFT     = kwargs.get('nFFT',None)
        self.__dt       = kwargs.get('sampPeriod',None)
        self.__nHop     = kwargs.get('nHop',None)
        self.__windowFn = kwargs.get('windowFn',None)
        self.__S        = kwargs.get('spectra',None)
        self.__fRange   = kwargs.get('freqRange',None)
        self.__coords   = kwargs.get('coordinates',None)
        self.__ID       = kwargs.get('ID',None)
        self.__t0       = kwargs.get('startTime',0)
        
        
        timeseries = kwargs.get('timeseries',None)
        
            
        # Init: spectrogram from time series
        if timeseries is not None:
            
            # Ensure that timeseries is an ArraySig object
            if not isinstance(timeseries,ArraySig):
                print('Timeseries data is not of class ArraySig')
                return
                
            # Overwrite dt with sampling period from 
            dt = timeseries.sampPeriod()
            self.__dt = dt
            
            # Overwrite coordiantes and IDs
            self.__coords = timeseries.getCoords()
            self.__ID = timeseries.getIDs()
            
            nFFT = self.__nFFT
            nHop = self.__nHop
            
            windowFn = self.__windowFn
            if hasattr(windowFn, '__call__'):
                # Define window (requires 'windows' module)
                windowFn = windowFn(nFFT)
            
            fRange = self.__fRange
            
            S = self.computeSpectrogram(timeseries,nFFT,nHop,dt,windowFn,fRange)
            
            self.__S = S[:,0:nFFT/2] # only store positive frequencies
            self.dt = dt
            self.nFFT = nFFT
            self.nHop = nHop
            self.windowFn = windowFn
            self.nHop = nHop
            return
            
        # If no timeseries are provided, then check sanity of inputs
        if self.__S is not None:
            '''
            Make sure dimensions add up (number of coordinates and ID and frequencies match with self.__S)
            '''
            pass
        
        
        
    def getCohMatrix(self,f0,M,tidx):
        '''
        Return a sample coherence matrix based on M snapshots starting at 
        time index tidx. The result is delivered in a (N,N) numpy.array

        INPUT
        f0:     frequency of interest
        M:      Number of snapshots to return
        tidx:   starting time index
        '''
   
        # Find the correct frequency index
        f = self.getFreqVec()
        idx = np.where(f>=f0)
        if len(idx[0])>0:
            idx = idx[0][0]
        else:
            print('Frequency not available.')
            return None

        S = self.__S[tidx:tidx+M-1,idx,:].T
        S = S/np.abs(S)
        return (1/float(M)) * S.dot(np.conj(S.T))        
    
    def getCovMatrix(self,f0,M,tidx):
        '''
        Return a sample covariance matrix based on M snapshots starting at 
        time index tidx. The result is delivered in a (N,N) numpy.array

        INPUT
        f0:     frequency of interest
        M:      Number of snapshots to return
        tidx:   starting time index
        '''
   
        # Find the correct frequency index
        f = self.getFreqVec()
        idx = np.where(f>=f0)
        if len(idx[0])>0:
            idx = idx[0][0]
        else:
            print('Frequency not available.')
            return None

        S = self.__S[tidx:tidx+M-1,idx,:].T
        return (1/float(M)) * S.dot(np.conj(S.T))
    
    
    def getNumSamp(self):
        return self.__S.shape[0]
        
        
    def getNumSignals(self):
        return self.__S.shape[2]
        
        
    def getHop(self):
        return self.__nHop
        
        
    def getFreqVec(self):
        fVec = spectrogram.getFreqVec(self.__nFFT,1/self.__dt)
        if self.__fRange is None:
            return fVec
        else:
            return fVec[np.logical_and(fVec>=self.__fRange[0],fVec<=self.__fRange[1])]
        
    def getSpectra(self,*args):
        
        
        if len(args)==0:
            return self.__S
        elif len(args)==1 and isinstance(args[0],int): # and args[0]>=0 and args[0]<=self.__data.shape[1]
            
            # Create a spectrogram obect
            return spectrogram.Spectrogram(spectra=self.__S[:,:,args[0]],nFFT=self.nFFT,nHop=self.nHop,windowFn=self.windowFn,dt=self.dt)
        
        
            # self.__S = S
#             self.dt = dt
#             self.nFFT = nFFT
#             self.nHop = nHop
#             self.windowFn = windowFn
                    
        
            # return self.__S[:,:,args[0]]
        else:
            print('Selector must be integer identifying signal')
            return
        
        
    def computeSpectrogram(self,timeseries,nFFT,nHop,dt,windowFn,fRange):
        '''
        Compute STFT spectrogram of data in timeseries. Expected numpy.array shape
        (nSamples,nSignals)
        
        Returns: spectra, (nTimes,nFreq,nSignals)
        '''
        
        data = timeseries.getData()
        
        fVec = self.getFreqVec()
        if fRange is not None:
            mask = np.logical_and(fVec>=fRange[0], fVec<=fRange[1])
        else:
            # Create an all true array
            mask = np.ones((len(fVec),1))==1
            
        mask = np.nonzero(mask)
        mask = mask[0]
        
        # Loop over signals (2nd dimension of )
        Nsignals = timeseries.numSignals()
        for i in range(Nsignals):
            x = data[:,i,np.newaxis]
            X = spectrogram.stft(x, nFFT, nHop, transform=np.fft.fft, win=windowFn, zp_back=0, zp_front=0)
            
            if i==0:
                spectra = np.zeros((X.shape[0],len(mask),Nsignals),dtype=complex)
            
            spectra[:,:,i] = X[:,mask]
        
        return spectra
    
    
    def blockAverage(self,M,step):
        
        
        # Loop over spectrograms and block average them
        
        
        for i in range(self.__S.shape[2]):
            Si = spectrogram.blockAverage(self.__S[:,:,i],M,step)
            
            if i == 0:
                S = np.zeros((Si.shape[0],Si.shape[1],self.__S.shape[2]),dtype=complex)
                
            S[:,:,i] = Si
            
        
        self.__S = S
