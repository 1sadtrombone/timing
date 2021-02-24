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

    print("incoming PFBed data")
    print(data.shape)
    
    # zfill in chunks
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
    print("ffted")
    print(ffted.shape)

    # pick out orbcomm channs

    # boxcar:
    #ffted[:fi] = 0
    #ffted[ff:] = 0
    
    return ffted    

if __name__ == "__main__":

    # get PFB'ed data
    print(f"working on: {fname}")
    #header, data = albatrostools.get_data(f"{data_dir}{fname}", items=-1, unpack_fast=True, float=True)
    # need to use the old read 4bit code (direct grumbles to Nivek
    pol0, pol1 = read_4bit.read_4bit_new(f"{data_dir}{fname}")
    print("unpacked")

    skip = 0
    n_samples = 40*1000

    pol0 = pol0[skip:n_samples+skip]
    pol1 = pol1[skip:n_samples+skip]

    chans = read_4bit.read_header(f"{data_dir}{fname}")['chans']

    # eyeballed, second peak from left (seems clearly defined)
    #fi = int(1.5006e7)
    #ff = int(1.5034e7)

    # odd, it seems to have drastically shifted... chose third from left this time
    fi = int(7.531e6)
    ff = int(7.543e6)

    ipfb_chunk = 4*1000
    ifft_chunk = 2**12

    # trim to integer chunk count
    pol0 = pol0[:n_samples-n_samples%ifft_chunk]
    pol1 = pol1[:n_samples-n_samples%ifft_chunk]

    print('pol0 trimmed again')
    print(pol0.shape)
    
    corr = np.zeros(ifft_chunk*2048+1, dtype=np.complex128)

    print('corr created')
    print(corr.shape)

    for i in range(pol0.shape[0]//ifft_chunk):
        print(f'iteration {i}')

        pol0_fft = get_isolated_ffts(pol0[i*ifft_chunk:(i+1)*ifft_chunk], chans, fi, ff, N=ipfb_chunk)
        pol1_fft = get_isolated_ffts(pol1[i*ifft_chunk:(i+1)*ifft_chunk], chans, fi, ff, N=ipfb_chunk)

        add = pol0_fft*np.conj(pol1_fft)
        corr += add

    """
    # gaussian:
    fwhm = ff - fi
    mu = (ff + fi)/2
    sig = fwhm / (2*np.sqrt(2*np.log(2)))
    xs = np.arange(ffted.size)
    gauss_window = 1/(sig*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sig**2))

    ffted = ffted * gauss_window
    """
    """
    # centre the peak
    N = phase.size
    phase = np.hstack((phase[N//2:], phase[:N//2]))
    dt = 1/(250e6)
    ts = np.linspace(0,N*dt,N) - N/2*dt
    """

    name = f"data/xcorr_lab_fspace"
    print(f"saving data at {name}")
    np.save(name, corr)
