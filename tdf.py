import numpy as np
from scipy import constants, signal, fft

def rtdf(f, y, t1, t2 = 1e10, w_f = np.array(1), w_t = np.array(0), offset = 0, b_remove_dc = True):
    """ Applies a time-domain filter to a given real-valued spectral dataset 'y'.

    Parameters
    ----------
    f: 1-D array
        Frequencies of the corresponding spectral data.
        
    y: 1-D array or 2-D array
        Spectral data. For a 2-D array, the rows must correspond to the different spectra and the columns to the frequencies.
    
    t1: float
        Cut-off value of the time-domain filter starts in the time-domain.
    
    t2: float
        Cut-off value of the time-domain filter stops in the time-domain.
    
    w_f: 1-D array
        Spectral filter applied to the input spectral data before time-domain filtering.

    w_f: 1-D array
        Filter applied to the time-domain. If specified, this is used instead of t1 and t2.

    offset: string or float
        offset = 'mean': Adds mean value of the original spectrum to the spectrum after time-domain filtering.
        float: Adds a specific value to the spectrum after time-domain filtering.

    b_remove_dc: boolean
        True/False: Remove DC value if the input spectra   

    
    Returns:
    ----------    
    y_f : 1-D or 2-D array, same shape as 'y'
        The filtered data only with the frequency filter applied.

    y_ff : : 1-D or 2-D array, same shape as 'y'
        Time-domain filtered data.

    t : 1-D array
        The time axis of the time domain representation of the input spectra.

    At : 1-D or 2-D array
        The time-domain representation of the input spectra.

    At_f : 1-D or 2-D array
        The time-domain representation of the input spectra with applied time-domain filter.

    w_f : 1-D array
        The applied frequency filter.

    w_t : 1-D array
        The applied time-domain filter
    """
    assert np.iscomplexobj(y) == False, 'real-valued input required'
    
    # calculate time-axis
    t = fft.rfftfreq(f.size, d=np.diff(f*constants.c*100).mean())*1E15
    
    ## define filter functions in the time- and frequency domain
    # consider entire frequency range, if no frequency filter function is provided
    w_f = np.ones_like(f) if np.equal(1, w_f).all() else w_f
    
    # use Heaviside step function, if no custom temporal filter function is provided
    if np.equal(0, w_t).all():
        w_t = np.zeros(t.size)
        t1, t2   = (np.abs(t - t1)).argmin(), (np.abs(t  - t2)).argmin()
        w_t[t1:t2] = 1
    else:
        w_t
    
    # converts the input array into a 2D array if it is given as 1D
    if y.ndim == 1:
        y = y.reshape((1,y.size))
        b_1d = True
    else:
        b_1d = False
    
    ## process FD prior to FF
    y_f = y
    # center the average of the input around 0
    y_f_mean = np.average(y_f, axis=1, weights=w_f) 
    if b_remove_dc == True:
        y_f = y_f - y_f_mean[:,None] 
        
    y_f = y_f*w_f[None,:] # apply FD filter
    
    # perform the actual time-domain filtering
    At = fft.rfft(y_f, axis=1)
    At_f = At*w_t[None,:] # apply TD filter
    y_ff = fft.irfft(At_f, axis=1, n=f.size) # make sure the output has the same dimensions as the input
    
    # add mean again to obtain td-filtered spectrum
    if b_remove_dc == True:
        y_f  = y_f  + y_f_mean[:,None]

    # add mean if input was centered around 0 before performing the FF
    if offset == 'mean':
        y_ff = y_ff + y_f_mean[:,None]
    else:
        y_ff = y_ff + offset
    
    # convert back to 1D if the input was 1D
    if b_1d:
        y_f  = np.ravel(y_f)
        y_ff = np.ravel(y_ff)
        At   = np.ravel(At)
        At_f = np.ravel(At_f)
  
    return y_f, y_ff, t, At/At.max(), At_f/At.max(), w_f, w_t 

def tdf(f, y, t1, t2 = 1e10, w_f = np.array(1), w_t = np.array(0), offset = 0):
    """ Applies a time-domain filter to a given spectral dataset 'y'. Thereby the real and imaginary part of 'y' are treated independently.

    Parameters
    ----------
    f: 1-D array
        Frequencies of the corresponding spectral data.
        
    y: 1-D array or 2-D array
        Spectral data. For a 2-D array, the rows must correspond to the different spectra and the columns to the frequencies.
    
    t1: float
        Cut-off value of the time-domain filter starts in the time-domain.
    
    t2: float
        Cut-off value of the time-domain filter stops in the time-domain.
    
    w_f: 1-D array
        Spectral filter applied to the input spectral data before time-domain filtering.

    w_f: 1-D array
        Filter applied to the time-domain. If specified, this is used instead of t1 and t2.

    offset: float
        float: Adds a specific value to the spectrum after time-domain filtering.

    Returns:
    ----------    
    y_ff : : 1-D or 2-D array, same shape as 'y'
        Time-domain filtered data.
    """
    if np.iscomplexobj(y):
        y_ff_real = rtdf(f, y.real, t1, t2 = t2, w_f = w_f, w_t = w_t, offset = offset.real)[1]
        y_ff_imag = rtdf(f, y.imag, t1, t2 = t2, w_f = w_f, w_t = w_t, offset = offset.imag)[1]
        y_ff = y_ff_real + 1j*y_ff_imag
    else:
        y_ff = rtdf(f, y, t1, t2 = t2, w_f = w_f, w_t = w_t, offset = offset)[1]
    return y_ff

def ctdf(f, y, t1, t2, w_f = np.array(1), w_t = np.array(0), b_remove_dc = True, offset=0):
    """ Applies a time-domain filter to a given complex-valued spectral dataset 'y'. The real and imaginary part are treated together.

    Parameters
    ----------
    f: 1-D array
        Frequencies of the corresponding spectral data.
        
    y: 1-D array or 2-D array
        Spectral data. For a 2-D array, the rows must correspond to the different spectra and the columns to the frequencies.
    
    t1: float
        Cut-off value of the time-domain filter starts in the time-domain.
    
    t2: float
        Cut-off value of the time-domain filter stops in the time-domain.
    
    w_f: 1-D array
        Spectral filter applied to the input spectral data before time-domain filtering.

    w_f: 1-D array
        Filter applied to the time-domain. If specified, this is used instead of t1 and t2.

    offset: string or float
        offset = 'mean': Adds mean value of the original spectrum to the spectrum after time-domain filtering.
        float: Adds a specific value to the spectrum after time-domain filtering.

    b_remove_dc: boolean
        True/False: Remove DC value if the input spectra   

    
    Returns:
    ----------    
    y_f : 1-D or 2-D array, same shape as 'y'
        The filtered data only with the frequency filter applied.

    y_ff : : 1-D or 2-D array, same shape as 'y'
        Time-domain filtered data.

    t : 1-D array
        The time axis of the time domain representation of the input spectra.

    At : 1-D or 2-D array
        The time-domain representation of the input spectra.

    At_f : 1-D or 2-D array
        The time-domain representation of the input spectra with applied time-domain filter.

    w_f : 1-D array
        The applied frequency filter.

    w_t : 1-D array
        The applied time-domain filter
    """
    # time axis
    t = fft.fftshift(fft.fftfreq(f.size, d=np.diff(f*constants.c*100).mean()))*1E15 # calculated time-axis
    
    # filter functions    
    w_f = np.ones_like(f) if np.equal(1, w_f).all() else w_f

    # use Heaviside step function, if no custom temporal filter function is provided
    if np.equal(0, w_t).all():
        w_t = np.zeros(t.size)
        t1, t2   = (np.abs(t - t1)).argmin(), (np.abs(t  - t2)).argmin()
        w_t[t1:t2] = 1
    else:
        w_t
    
    # converts the input array into a 2D array if it is given as 1D
    if y.ndim == 1:
        y = y.reshape((1,y.size))
        b_1d = True
    else:
        b_1d = False

    ## process FD prior to FF
    y_f = y
    # center the average of the input around 0
    y_f_mean = np.average(y_f, axis=1, weights=w_f) 
    if b_remove_dc == True:
        y_f = y_f - y_f_mean[:,None] 
        
    y_f = y_f*w_f[None,:] # apply FD filter

    # perform the actual time-domain filtering
    At = fft.fftshift(fft.ifft(y_f,axis=1),axes=1)
    At_f = At*w_t[None,:] # apply TD filter
    y_ff = fft.fft(fft.ifftshift(At_f,axes=1),axis=1, n=f.size)

    # add mean again to obtain td-filtered spectrum
    if b_remove_dc == True:
        y_f  = y_f  + y_f_mean[:,None]

    # add mean if input was centered around 0 before performing the FF
    if offset == 'mean':
        y_ff = y_ff + y_f_mean[:,None]
    else:
        y_ff = y_ff + offset    
    
        # convert back to 1D if the input was 1D
    if b_1d:
        y_f  = np.ravel(y_f)
        y_ff = np.ravel(y_ff)
        At   = np.ravel(At)
        At_f = np.ravel(At_f)

    return y_f, y_ff, t, At/At.max(), At_f/At.max(), w_f, w_t 

def tukeyF(t,t1,t2,t_width):
    """
    Creates a Tukey-like bandpass filter for a given input axis t.
    The Tukey-filter is 1 between t1 and t2 and drops to 0 with a transition width of t_width.

    Parameters
    ----------
    t: 1-D array
        Input axis.
    
    t1: float
        Lower value of the band-pass filter.
    
    t2: float
        Higher value of the band-pass filter.
    
    t_width: float
        Transition length of the filter.

    Returns:
    ----------    
    window_ t : 1-D , same shape as 't'
        Returns the Tukey-shaped bandpass filter.
    """
    idx_t1, idx_t2 = (np.abs(t - t1)).argmin(), (np.abs(t - t2)).argmin()
    idx_t_width = (np.abs(t-t[0]-t_width)).argmin()
    N = idx_t2 - idx_t1 + 1 + 2*idx_t_width
    alpha = 2*idx_t_width/N
    window_t = np.zeros(t.size)
    
    idx_len = -max(0,idx_t1-idx_t_width)+min(t.size,idx_t2+idx_t_width)
    window_t[max(0,idx_t1-idx_t_width):min(t.size,idx_t2+idx_t_width)] =(
            signal.tukey(N,alpha)[max(0,-(idx_t1-idx_t_width)):max(0,-(idx_t1-idx_t_width))+idx_len])
    return window_t

