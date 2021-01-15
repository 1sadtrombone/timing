import numpy as np
import matplotlib.pyplot as plt
import albatrostools
import read_4bit
import pfb_helper as pfb

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
    
    - data: baseband data (which is PFBed). Works best if length is power of small primes
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
        zfilled = np.zeros((min(N, data.shape[0]-i*N),2049), dtype=np.complex64)
        if zfilled.shape[0] < ntap:
            continue

        zfilled[:,chans]=data[i*N:(i+1)*N]
        spec = inverse_pfb_fft_filt(zfilled, ntap, thresh=0.1)
        ts[i*N*un_raveled_chans:(i+1)*N*un_raveled_chans] = np.ravel(spec)

    # IPFB -> FFT
    ffted = np.fft.rfft(ts)/ts.size
    
    # pick out orbcomm channs
    ffted[:fi] = 0
    ffted[ff:] = 0

    return ffted

def get_corr(timestreams, chans, fi, ff, ipfb_chunk):
    """
    Get timestream correlation of the two timestreams.
    
    -timestreams: a tuple of timestream data. Should be same length and from same frequencies.
    -chans: channels which the timestreams are from. Should be in the header of the beaseband files.
    -fi/ff: initial/final frequency to consider, in Hz. All else is zeroed before correlation.
    -ipfb_chunk: chunk size for the IPFB computation. Should be a multiple of ntap (4)
    """

    return np.fft.irfft(get_isolated_ffts(timestreams[0], chans, fi, ff, N=ipfb_chunk)*np.conj(get_isolated_ffts(timestreams[1], chans, fi, ff, N=ipfb_chunk)))
    

if __name__ == "__main__":

    # get PFB'ed data
    print(f"working on: {fname}")
    #header, data = albatrostools.get_data(f"{data_dir}{fname}", items=-1, unpack_fast=True, float=True)
    # need to use the old read 4bit code (direct grumbles to Nivek)
    pol0, pol1 = read_4bit.read_4bit_new(f"{data_dir}{fname}")
    print("unpacked")

    skip = 0
    n_samples = 40*1000

    pol0 = pol0[skip:n_samples+skip]
    pol1 = pol1[skip:n_samples+skip]

    chans = read_4bit.read_header(f"{data_dir}{fname}")['chans']

    # eyeballed, second peak from left (seems clearly defined)
    fi = int(1.5006e7)
    ff = int(1.5034e7)

    ipfb_chunk = 4*1000
    ifft_chunk = 2**13

    N = pol0.shape[0]

    # trim to integer chunk count
    pol0 = pol0[:N-N%ifft_chunk]
    pol1 = pol1[:N-N%ifft_chunk]
    
    corr = np.zeros(pol0.size)

    for i in range(pol0.shape[0]//ifft_chunk):
        corr += get_corr((pol0[i*ifft_chunk:(i+1)*ifft_chunk], pol1[i*ifft_chunk:(i+1)*ifft_chunk]), chans, fi, ff, ipfb_chunk)
    
    # centre the peak
    N = corr.size
    corr = np.hstack((corr[N//2:], corr[:N//2]))
    dt = 1/(250e6)
    ts = np.linspace(0,N*dt,N) - N/2*dt

    name = f"data/xcorr_lab_{n_samples}samples_agg"
    print(f"saving data at {name}")
    np.save(name, corr)
