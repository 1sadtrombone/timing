import numpy as np
import matplotlib.pyplot as plt
import albatrostools
import read_4bit
import pfb_helper as pfb
from multiprocessing import Pool, set_start_method, get_context

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

fname = "935278854.raw"
data_dir = "/project/s/sievers/sievers/albatros/lab_baseband/orbcomm/"

#corrs = albatrostools.correlate(data['pol0'], data['pol1'])
#corrs = albatrostools.bin_crosses(data['pol0'], data['pol1'], chunk=4096)

def inverse_pfb_fft_filt(dat,ntap,window=pfb.sinc_hamming,thresh=0.0):
    dd=np.fft.irfft(dat,axis=1)
    win=window(ntap,dd.shape[1])
    win=np.reshape(win,[ntap,len(win)//ntap])
    mat=np.zeros(dd.shape,dtype=dd.dtype)
    mat[:ntap,:]=win
    matft=np.fft.rfft(mat,axis=0)
    ddft=np.fft.rfft(dd,axis=0)
    if thresh>0:
        filt=np.abs(matft)**2/(thresh**2+np.abs(matft)**2)*(1+thresh**2)
        ddft=ddft*filt
    return np.fft.irfft(ddft/np.conj(matft),axis=0)

def get_isolated_ffts(data, chans, fi, ff, N=20000):
    """
    Get timestream of data if signal from fi to ff only is nonzero
    
    - data: baseband data (which is PFBed)
    - chans: frequency channels associated with this data
    - fi, ff: initial and final frequencies of the non-zero signal
    - N: chunk size for the IPFB computation
    """

    un_raveled_chans = 4096
    ntap = 4
    ts = np.zeros(data.shape[0]*un_raveled_chans, dtype=np.float64)    

    # work in chunks
    for i in range(data.shape[0]//N+1):
        # zfill unmentioned channels

        zfilled = np.zeros((min(N, pol0.shape[0]-i*N),2049), dtype=np.complex64)
        print(zfilled[:,chans].shape, pol0[i*N:(i+1)*N].shape)
        if zfilled.shape[0] < ntap:
            continue
        zfilled[:,chans]=pol0[i*N:(i+1)*N]
        spec = inverse_pfb_fft_filt(zfilled, ntap, thresh=0.1)
        ts[i*N*un_raveled_chans:(i+1)*N*un_raveled_chans] = np.ravel(spec)

    # IPFB -> FFT
    print(f"timestream length before trim: {ts.size}")
    trim = 15000
    ts = ts[trim:ts.size-trim]
    # for optimal speed, get this length to be a power of small primes
    n = int(np.log2(ts.size))
    ts = ts[:2**n]
    print(f"timestream length after trim: {ts.size} (2**{n})")
    
    ffted = np.fft.rfft(ts)/ts.size
    
    # pick out orbcomm channs
    ffted[:fi] = 0
    ffted[ff:] = 0

    return ffted

if __name__ == "__main__":

    # get PFB'ed data
    print(f"working on: {fname}")
    #header, data = albatrostools.get_data(f"{data_dir}{fname}", items=-1, unpack_fast=True, float=True)
    # need to use the old read 4bit code (direct grumbles to Nivek)
    pol0, pol1 = read_4bit.read_4bit_new(f"{data_dir}{fname}")
    print("unpacked")

    skip = 0
    n_samples = 1001004

    pol0 = pol0[skip:n_samples+skip]
    pol1 = pol1[skip:n_samples+skip]

    chans = read_4bit.read_header(f"{data_dir}{fname}")['chans']

    # eyeballed, second peak from left (seems clearly defined)
    fi = int(1.5006e7)
    ff = int(1.5034e7)

    chunk_n = 100100
    
    # correlate the pols
    corr = np.fft.irfft(get_isolated_ffts(pol0, chans, fi, ff, N=chunk_n)*np.conj(get_isolated_ffts(pol1, chans, fi, ff, N=chunk_n)))
    # centre the peak
    N = corr.size
    corr = np.hstack((corr[N//2:], corr[:N//2]))
    dt = 1/(250e6)
    ts = np.linspace(0,N*dt,N) - N/2*dt

    np.save(f"$SCRATCH/timing_data/xcorr_lab_{n_samples}samples_{skip}skip", corr)
