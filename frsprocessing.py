import numpy as np
import pandas as pd
import json
import tdf

from scipy import signal, constants

### scripts related to Figure 1-4 of the paper ###

def frs_fft(t,td_ref,td_sam, noecho=False):
    """
    Converts measured pairs (sample and reference) of raw FRS time-domain traces to complex sample transfer functions.

    Parameters
    ----------
    t: 1-D array
        Input time axis
    
    td_ref: 2-D array
        Reference traces. The first axis corresponds to different measurements. The second axis corresponds to the time axis.
    
    td_sam: 2-D array
        Sample traces. The first axis corresponds to different measurements. The second axis corresponds to the time axis.
    
    noecho: boolean
        True: Filters out a multiple reflection in the raw time traces.

    Returns:
    ----------    
    t : 1-D , same shape as 't'
        Returns the input time axis.

    td_ref : 2-D array, same shape as 'td_ref'
        Returns the input reference time-domain data. Pulse echo is removed if noecho = True.

    td_sam : 2-D array, same shape as 'td_sam'
        Returns the input sample time-domain data. Pulse echo is removed if noecho = True.

    freq : 1-D array
        Returns the frequency axis of the sample transfer function.

    hc : 2-D array
        Return the complexsample transfer functions.
        The first axis corresponds to different measurements. The second axis corresponds to the frequency axis.
    """
    dt=t[-1]-t[-2] # time resolution in fs
    Nt=len(t) # numer of samplnig points
    
    if noecho: # remove pulse echo - cut-off is hard-coded
        idx_thptf = (np.abs(t - 1450)).argmin()
        window_t = np.zeros(t.shape)
        window_t[:idx_thptf] = signal.windows.tukey(idx_thptf,0.1) # define Tukey filter
        td_ref, td_sam = td_ref*window_t, td_sam*window_t

    # calculate complex transfer function
    fdc_ref = np.fft.rfft(td_ref) # reference spectra
    fdc_sam = np.fft.rfft(td_sam) # sample spectra
    freq = np.fft.rfftfreq(Nt,d=dt*1E-15)/1E12/2.99792E-8/1E6 # frequency axis in wavenumbers
    hc = (fdc_sam)/(fdc_ref)
    
    return t, td_ref,td_sam, freq, hc

    # returns averaged spectra for a given conc and meas day

def meanspectra(metafile, seq, conc, t, freq, td_ref,td_sam, t1 = 500, t2=1400):
    """
    Calculates the averaged data (spectra, time-domain traces, etc.) for a given measured concentration of DMSO2 and measurement day.

    Parameters
    ----------
    metafile: pandas Dataframe
        Contains meta-information about the measurements.
    
    seq: int
        Select the sequence (equal to the measurment day) that should be considered.

    conc: int
        Select the concentration of DMSO2 that should be considered. 

    t: 1-D array
        Input time axis.
    
    freq : 1-D array
        Corresponding frequency axis of 't'.
    
    td_ref: 2-D array
        Reference traces. The first axis corresponds to different measurements. The second axis corresponds to the time axis.
    
    td_sam: 2-D array
        Sample traces. The first axis corresponds to different measurements. The second axis corresponds to the time axis.

    t1: float
        Lower value of the band-pass filter that is applied in the time-domain.
    
    t2: float
        Higher value of the band-pass filter that is applied in the time-domain.    

    Returns:
    ----------    
    td_ref_mean : 1-D array
        Averaged reference traces.
    td_sam_mean : 1-D array
        Averaged sample traces
    td_diff_mean : 1-D array
        Averaged time-domain difference traces.
    td_hc : 1-D array
        Averaged time-domain representation of the sample transfer function.
    td_hc_tdf : 1-D array
        Averaged time-domain representation of the time-domain sample transfer function.
    fdc_ref : 1-D array
        Averaged reference spectra.
    fdc_sam : 1-D array
        Averaged sample spectra.
    fdc_diff : 1-D array
        Averaged spectra of time-domain differences.
    fdc_diff_tdf : 1-D array
        Averaged spectra of time-filtered time-domain differences.
    hc : 1-D array
        Averaged spectra of sample transfer functions.
    hc_tdf : 1-D array
        Averaged spectra of time-domain filtered sample transfer functions.
    """
    index = metafile[(metafile["sequence ID"] == seq) & (metafile["Conc [ug/ml]"] == conc)].index.to_numpy()
    td_ref_mean = np.mean(td_ref[index],axis=0)
    td_sam_mean = np.mean(td_sam[index],axis=0)
    td_diff_mean = td_ref_mean - td_sam_mean
    fdc_ref = np.fft.rfft(td_ref_mean)
    fdc_sam = np.fft.rfft(td_sam_mean)
    fdc_diff = np.fft.rfft(td_diff_mean)
    fdc_diff_tdf = np.fft.rfft(td_diff_mean*tdf.tukeyF(t,t1,t2,50))
    hc = fdc_sam/fdc_ref
    w_f=tdf.tukeyF(freq,1055,1355,125)
    t_c = np.fft.fftshift(np.fft.fftfreq(freq.size, d=np.diff(freq*constants.c*100).mean()))*1E15
    _,_,_, td_hc,_,_,_ = tdf.ctdf(freq, hc, -500, 5000, w_t=tdf.tukeyF(t_c,t1,t2,50), w_f=w_f)
    _,hc_tdf,_, _,td_hc_tdf,_,_ = tdf.ctdf(freq, hc, t1, t2, w_t=tdf.tukeyF(t_c,t1,t2,50), w_f=w_f)
    return td_ref_mean, td_sam_mean, td_diff_mean, td_hc, td_hc_tdf, fdc_ref, fdc_sam, fdc_diff, fdc_diff_tdf, hc, hc_tdf


def concdot(hc,hc_Fit,idx_f1,idx_f2, baseline = 1):
    """
    Calculates the dot-product between two input vectors.
    Under the assumption that both vectors have the same shape, it can be used for concentration estimation of the measured spectrum.

    Parameters
    ----------
    hc: 1-D array
        Measured spectrum of a known substance with unknown concentration.

    hc: 1-D array
        Reference spectrum of the same substance with known concentration.       

    idx_f1,idx_f2: int, int
        Range within the dot-product will be evaluated.

    baseline: float, optional
        Option to remove a DC offset before calculation of the dot-product.

    Returns:
    ----------    
    c : float
        The dot-product between the two vectors. 'c' is equivalent to the relative concentration of the two input spectra.
    """
    return np.real(np.vdot((hc_Fit[idx_f1:idx_f2]-baseline),(hc[idx_f1:idx_f2]-baseline)))/np.linalg.norm(hc_Fit[idx_f1:idx_f2]-baseline)**2

def pathlengthcorrection(metafile, hc, hc_tdf,idx_start,idx_stop,seq, corr=1):
    """
    Normalizes the measured complex sample transfer functions to a specified length of the measuremen cuvette.

    Parameters
    ----------
    metafile: pandas Dataframe
        Contains meta-information about the measurements.

    hc: 2-D array
        Spectral dataset.

    hc_tdf: 2-D array
        Time-domain filtered spectral dataset.       

    idx_start,idx_stop: int, int
        Range that is considered when calculating the correction.

    seq: int
        Measurement sequence/day to which the path length of the other measurements is scaled.

    corr: 1 or list/array
        Custom scaling of the path lengths.

    Returns:
    ----------    
    hc : 2-D array
        Pathlenght corrected input spectral dataset.
    hc_tdf : 2-D array
        Pathlenght corrected input time-domain filtered spectral dataset.
    mu_corr : array
        Correction factor of the individual measurement days.
    """
    # select reference spectra to which the pathlength of the others will be scaled to
    index = metafile[(metafile["sequence ID"] == seq) & (metafile["Conc [ug/ml]"] == 1000.0)].index.to_numpy()
    hc_tdf_fit = np.mean(hc_tdf[index,:], axis=0)

    mu_corr = [] # correction factor
    for j, seq in enumerate(metafile["sequence ID"].unique()): # loop through all measurement days
        # selecet all TF of 1000ug/ml one sequence
        filter_conc = metafile["Conc [ug/ml]"] == 1000.0
        index = metafile[(metafile["sequence ID"] == seq) & filter_conc].index.to_numpy()
        # use 1000ug/ml to calculate correction factor
        if np.equal(1, corr).all() == True:
            # Here, we use the assumption that all 1000ug/ml measurements should have the same signal strength.
            # The calculated relative signal strength of these measurements can be used to correct also the measurements of lower concentrations.
            c = concdot(np.mean(hc_tdf[index,:], axis=0),hc_tdf_fit,idx_start,idx_stop)
        else:
            c = corr[j]
        # selecet all sample transfer function of one sequence and apply the correction
        index = metafile[(metafile["sequence ID"] == seq)].index.to_numpy()
        hc_tdf[index,:] = (hc_tdf[index,:]-1)/c+1 # apply correction
        hc[index,:] = (hc[index,:]-1)/c+1 # apply correction
        mu_corr.append(c)
    return hc, hc_tdf, mu_corr


def calc_diff(metafile,td_ref, td_sam, t, t1, t2, t_width, corr):
    """
    Apply path lenght correction to time-domain difference data.
    """
    td_diff = (td_ref-td_sam)/(np.sum(np.abs(td_ref), axis=1))[:,None]

    for j, seq in enumerate(metafile["sequence ID"].unique()):
        index = metafile[(metafile["sequence ID"] == seq)].index.to_numpy()
        td_diff[index,:] = td_diff[index,:]/corr[j] # apply correction

    return td_diff, np.fft.rfft(td_diff*tdf.tukeyF(t,t1,t2,t_width))
    
### scripts related to Figure 5 and 6 of the paper ###


def batch_readCSVspectra(paths):
    """
    Load FRS spectra saved as csv files.

    Parameters
    ----------
    paths: list of strings
        List of paths of the *.csv files

    Returns:
    ----------    
    freq : 1-D array
        Frequency axis.

    hc_out: 2-D array
        Complex sample transfer functions.
        The first axis corresponds to different measurements. The second axis corresponds to the frequency axis.
    """
    hc_out = []
    for path in paths:
        df = pd.read_csv(path, sep=',',header=None)
        freq = df[0].to_numpy()
        hc  = np.empty(freq.size, dtype='complex')
        hc  = df[1].to_numpy() + df[2].to_numpy()*1j
        hc_out.append(hc)
    return freq, np.vstack(hc_out)


def batch_readJSON(paths):
    """
    Load FTIR absorbance spectra saved as json files.

    Parameters
    ----------
    paths: list of strings
        List of paths of the *.json files

    Returns:
    ----------    
    freq : 1-D array
        Frequency axis.

    AB_out: 2-D array
        Absorbance spectra.
        The first axis corresponds to different measurements. The second axis corresponds to the frequency axis.
    """
    AB_out = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
            AB_out.append(data['AB Data'][::-1])
            wavenumber_start = data['Data Status Parameters']['LXV']
            wavenumber_end = data['Data Status Parameters']['FXV']
            wavenumber_inverval = data['Data Status Parameters']['NPT']
            freq = np.arange(wavenumber_start, wavenumber_end, (wavenumber_end - wavenumber_start) / wavenumber_inverval)   
    return freq, np.vstack(AB_out)


def interp_database_FTIR(freq_FRS, freq_FTIR, H_FTIR, df_FTIR):
    """
    Averages spectra of the same substance and performs an interpolation to the frequency grid of the FRS data.

    Parameters
    ----------
    freq_FRS: 1-D array
        Frequency axis of the FRS data.

    freq_FTIR: 1-D array
        Frequency axis of the FTIR data.

    H_FTIR: 2-D array
        Absorbance spectra.
        The first axis corresponds to different measurements. The second axis corresponds to the frequency axis.

    df_FTIR: pandas Dataframe
        Contains meta-information about the measurements.


    Returns:
    ----------    
    df_database_FTIR : pandas dataframe
        Contains meta-information about the spectral database.

    H_database_FTIR: 2-D array
        Absorbance spectra of the assembled data.
        The first axis corresponds to different measurements. The second axis corresponds to the frequency axis.
    """
    H_database_FTIR = []
    for substance in df_FTIR['substance'].unique():
        idx = df_FTIR[df_FTIR['substance'] == substance].index
        H = np.mean(H_FTIR[idx],axis=0)
        H = np.interp(freq_FRS,freq_FTIR,H)
        H_database_FTIR.append(H)
    H_database_FTIR = np.vstack(H_database_FTIR)
    df_database_FTIR = pd.DataFrame()
    df_database_FTIR['substance'] = df_FTIR['substance'].unique()
    df_database_FTIR['conc'] = 1
    return df_database_FTIR, H_database_FTIR

def H_dbsearchsinglespecies(H_in, H_database):
    """
    Calculates the cosine similarity between an input spectrum and each spectrum contained in a spectral database.

    Parameters
    ----------
    H_in: 1-D array
        Input spectrum.

    H_database: 2-D array
        The first axis corresponds to different spectra. The second axis corresponds to the frequency axis.

    Returns:
    ----------    
    cosine_sim : 1-D array
        Cosine similarity for each spectrum of the spectral database.

    """
    cosine_sim = []
    for _, H_ref in enumerate(H_database):
        # cosine similarity between H_in and H_ref
        cosine_sim.append((H_in@H_ref)**2/(np.linalg.norm(H_in)*np.linalg.norm(H_ref))**2) 
    cosine_sim = np.array(cosine_sim)
    return cosine_sim


def interp_dataset(x_out, x_in, y_in):
    """
    Interpolates a given spectral dataset to a specified frequency grid.

    Parameters
    ----------
    x_out: 1-D array
        Target frequency axis.

    x_out: 1-D array
        Input spectrum.

    y_in: 2-D array
        Input spectral dataset.
        The first axis corresponds to different spectra. The second axis corresponds to the frequency axis.

    Returns:
    ----------    
    y_out : 2-D array
        Interpolated dataset.

    """
    y_out = []
    for y in y_in:
        y = np.interp(x_out,x_in,y)
        y_out.append(y)
    return np.vstack(y_out)