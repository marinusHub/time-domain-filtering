### Data class for handling, processing, and analysing spectral data
### v.0.1
### 04.07.2024 by Marinus Huber

# required packages
import numpy as np
import copy
import tdf

from scipy import  signal


class SpectralData:
    '''Class for handling, processing, and analysing spectral data'''
    def __init__(self, freq = None, data = None, data_type = 'fdc'):
        """
        Initialize the SpectralData instance.
        
        Parameters:
        freq : 1-D Array
            Frequency of the corresponding spectral data

        data : 2-D Array
            Spectral data. The rows correspond to the different spectra. The columns correspond to the frequencies.

        data_type : string
            Data type of the specified data.
            'hc': The data are complex transfer functions
            'abs': The date are absorbance spectra       
        """
        self.freq = freq # frequency in wavenumbers
        self.data = data # spectral data
        self.data_type = data_type # data type of the stored data (e.g. hc for complex transfer function) 
        self.backup_data = {} # storage for a given pre-processing state of the spectra

    def __call__(self):
        return [self.freq,self.data]
    
    def __getitem__(self, index):
        return self.data[index]
    
    ### methods to convert data-type ###
    def hc_to_abs(self):
        '''Converts complex sample transfer functions to absorbance.
        
        Returns:
            - Self       
        '''
        if self.data_type == 'hc':
            self.data = -np.log10(np.abs(self.data)**2)
            self.data_type = 'abs'
        else:
            print('only applicable to hc data')
        return self
    
    
    ### data handling and subselection ###
    def f_to_idx(self,f):
        '''
        Returns:
            - Tuple of indeces corresponding to given input frequency
        '''
        return (np.abs(self.freq - f[0])).argmin(), (np.abs(self.freq - f[1])).argmin()+1

    def reduce(self, f = None, idx = None):
        '''
        Reduction of the data set along the frequency axis or by selecting the spectra to be considered.

        Returns:
            - Self       
        '''
        if idx != None:
            self.data = self.data[idx[0]:idx[1]]
        if f!= None:
            f_idx = self.f_to_idx(f)
            self.data = self.data[:,f_idx[0]:f_idx[1]]
            self.freq = self.freq[f_idx[0]:f_idx[1]]
        return self
    
    def backup(self,key = 0):
        '''
        Saves a backup of the (processed) Dataset.

        Returns:
            - Self       
        '''
        self.backup_data[key] = copy.deepcopy([self.freq, self.data, self.data_type])  
        return self
    
    def loadbackup(self,key = 0):
        '''
        Loads a backup of the (processed) Dataset.

        Returns:
            - Self       
        '''
        self.freq, self.data, self.data_type = self.backup_data[key]
        return self

    ### pre-processing options ###
    def savgol(self,window_length, polyorder, **kwargs):
        """ Apply a Savitzky-Golay filter to to each spectra of the dataset.

        Parameters
        ----------
        A detailed description of the parameters can be found at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html.
        
        ----------
        Returns:
            - Self       
        """
        if self.data_type == 'hc':
            y_ff_real = signal.savgol_filter(self.data.real-1, window_length, polyorder, axis=1, **kwargs)+1
            y_ff_imag = signal.savgol_filter(self.data.imag, window_length, polyorder, axis=1, **kwargs)
            self.data = y_ff_real + 1j*y_ff_imag
        elif self.data_type == 'abs':
            self.data = signal.savgol_filter(self.data, window_length, polyorder, axis=1, **kwargs)
        else:
            print('Preprocessing option not applicable')
        return self
    
    def linblcorr(self, f1,f2):
        """ Apply a linear baseline correction to each spectra of the dataset.

        Parameters
        ----------
        f1/f2: float
            Spectral range within the linear baseline correction will be applied.
        ----------
        Returns:
            - Self       
        """
        if self.data_type == 'hc':
            self.data=linbasecorr(self.freq,self.data-1,f1,f2)+1
        elif self.data_type == 'abs':
            self.data=linbasecorr(self.freq,self.data,f1,f2)
        else:
            print('baseline correction not applicable')
        return self
    
    def tdf(self,t1=-500, offset = 'expected', **kwargs):
        """ Apply a time-domain filtering to each spectra of the dataset. Thereby the real and imaginary part of 'y' are treated independently.

        Parameters
        ----------
        t1: float
            Cut-off value of the time-domain filter in the time-domain
        
        offset: optional or float
            Offset value of the time-domain filtered data that is added after filtering. By default time-domain filtering removes any DC value. If offset == 'expected', the DC-value of a unity function of the chosen spectral data type will be added.
        
        **kwargs:
        ----------
        t2: float
            Cut-off value of the time-domain filter stops in the time-domain.
    
        w_f: 1-D array
            Spectral filter applied to the input spectral data before time-domain filtering.

        w_f: 1-D array
            Filter applied to the time-domain. If specified, this is used instead of t1 and t2.    
        
        ----------
        Returns:
            - Self       
        """
        if self.data_type == 'hc':
            offset = 1 if offset == 'expected' else offset
            self.data = tdf.tdf(self.freq, self.data, t1=t1, offset = offset, **kwargs)
        elif self.data_type == 'abs':
            offset = 0 if offset == 'expected' else offset
            self.data = tdf.tdf(self.freq, self.data, t1=t1, offset = 0, **kwargs)
        else:
            print('Time-domain filtering not applicable to data type')
        return self

### pre-processing functions ###
   
def basecorrnorm(freq, hc, f1, f2):
    '''
    applies a linear baseline correction to a given set of spectra
    freq: frequency axis
    hc: spectra
    f1, f2: spectral range that should be considered for the correction
    '''
    f1, f2 = (np.abs(freq - f1)).argmin(), (np.abs(freq - f2)).argmin()
    A = np.vstack([np.ones(f2-f1)]).T
    if hc.ndim == 1:
        return hc
    else:
        y, s = 0, 0
        for j in range(hc.shape[0]):
            f = np.linalg.lstsq(A,hc[j,f1:f2].real,rcond=None)
            hc[j] = hc[j] - (f[0][0])
            y += f[0][0]
            s += np.linalg.norm(hc[j,f1:f2].real)
            hc[j] = hc[j] / np.linalg.norm(hc[j,f1:f2].real)
        y = y / hc.shape[0]
        s = s / hc.shape[0]
        return hc*s + y

def bl_corr_hc(freq, hc, f1, f2):
    '''
    corrects the offset of a complex transmission (sets average transmission to 1, and the average phase to 0)
    '''
    f1, f2 = (np.abs(freq - f1)).argmin(), (np.abs(freq - f2)).argmin()
    offset_real = np.mean(hc[:,f1:f2].real, axis=1)
    offset_imag = np.mean(hc[:,f1:f2].imag, axis=1)
    hc = hc - offset_real[:,None] + 1 - 1j*offset_imag[:,None]
    return hc

def linbasecorr(freq, hc, f1, f2):
    '''
    applies a linear baseline correction using a least squares fit
    '''
    f1, f2 = (np.abs(freq - f1)).argmin(), (np.abs(freq - f2)).argmin()
    A = np.vstack([freq[f1:f2], np.ones(f2-f1)]).T
    if hc.ndim == 1:
        f = np.linalg.lstsq(A,hc[f1:f2],rcond=None)
        return hc - (freq*f[0][0]+f[0][1])
    else:
        for j in range(hc.shape[0]):
            f = np.linalg.lstsq(A,hc[j,f1:f2],rcond=None)
            hc[j] = hc[j] - (freq*f[0][0]+f[0][1])
        return hc
    
def tukeyF(t,t1,t2,t_width):
    '''
    returns a Tukey filter for a given time axis t
    '''
    idx_t1, idx_t2 = (np.abs(t - t1)).argmin(), (np.abs(t - t2)).argmin()
    idx_t_width = (np.abs(t-t[0]-t_width)).argmin()
    N = idx_t2 - idx_t1 + 1 + 2*idx_t_width
    alpha = 2*idx_t_width/N
    window_t = np.zeros(t.size)
    
    idx_len = -max(0,idx_t1-idx_t_width)+min(t.size,idx_t2+idx_t_width)
    window_t[max(0,idx_t1-idx_t_width):min(t.size,idx_t2+idx_t_width)] =(
            signal.windows.tukey(N,alpha)[max(0,-(idx_t1-idx_t_width)):max(0,-(idx_t1-idx_t_width))+idx_len])
    return window_t