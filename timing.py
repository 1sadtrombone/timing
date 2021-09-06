import numpy as np
import matplotlib.pyplot as plt
import albatrostools
import read_4bit
import pfb_helper as pfb

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

# "DOCUMENTATION"
# Feed this script the location of two baseband streams
# It works in chunks on the pfb'ed baseband
# getting the fft of each chunk, for each of the two data files.
# It then saves the average of the ffts (across chunks, as a fct. of frequency)
# for each data file provided. Standard dev. is similarly saved.
# The saved .npy files are further processed and plotted in plot_timing.py

# The process may become more clear after looking at the code below

# note: here is not where any correlation happens
# the files are worked on at the same time for convenience,
# but no interaction is had between them.
# The signals are compared in the chi^2 calculation
# found in plot_timing.py


# ipfb function borrowed from scio
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

def get_zerofilled_ffts(data, chans, N=20000):
    """
    Get fft of pfb'ed baseband, while zerofilling all channels without signal in them (instead of simply ignoring them)
    
    - data: baseband data (which is PFBed). Works best if length is power of small primes (eg 2)
    - chans: frequency channels with baseband data in them
    - N: chunk size for the IPFB computation
    """

    un_raveled_chans = 4096
    ntap = 4
    ts = np.zeros(data.shape[0]*un_raveled_chans, dtype=np.float64)    

    print("incoming PFBed data")
    print(data.shape)
    
    # zfill in chunks
    for i in range(data.shape[0]//N+1):

        # zfill all unmentioned channels (total 2048, well 2049 for fft ease (ask Jon))
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
    
    return ffted    

if __name__ == "__main__":

    dataset = 'lab'
    print(dataset)
    
    name = f"NAME_YOUR_DESTINATION_FILE" # filename to save to
    print(f'eventually writing to {name} (not yet)')

    # get PFB'ed data
    # there are 3 datasets right now. The lab dataset is the only one I've really worked with
    # uapishka is extra complicated because the two stations save data taken at the same time
    # under different filenames

    if dataset == "lab":
        
        fname = "935278854.raw"
        data_dir = "/project/s/sievers/sievers/albatros/lab_baseband/orbcomm/"
        print(f"working on: {data_dir}{fname}")
        # need to use the old read 4bit code (direct grumbles to Nivek)
        sig0, sig1 = read_4bit.read_4bit_new(f"{data_dir}{fname}")
        header = read_4bit.read_header(f"{data_dir}{fname}")

    if dataset == "gault":

        fname = "16184/1618433016.raw"
        data_dir = "/project/s/sievers/tristanm/MontStHilaire/"

        print(f"working on: {data_dir}{fname}")
        header, data = albatrostools.get_data(f"{data_dir}{fname}", items=-1, unpack_fast=True, float=True)
        sig0 = data['pol0']
        sig1 = data['pol1']
        
    if dataset == "uapishka":
        # I haven't figured out the above extra uapishka complication
        fname_ant0 = "snap1/16274/1627429131.raw"
        fname_ant1 = "snap3/16272/1627271677.raw"
        data_dir = "/project/s/sievers/albatros/uapishka/baseband/"
        
        fnames = [fname_ant0, fname_ant1]
        ant_data = np.zeros((2,2), dtype=object)
        for i, fname in enumerate(fnames):
            print(f"working on: {data_dir}{fname}")
            header, data = albatrostools.get_data(f"{data_dir}{fname}", items=-1, unpack_fast=True, float=True)
            ant_data[i,0] = data['pol0']
            ant_data[i,1] = data['pol1']
            
        # choose which signals to compare (antenna, polarization)
        sig0 = ant_data[0,0]
        sig1 = ant_data[1,0]
    
    print("unpacked")
    
    # to have a quick look at the data:
    """
    plt.imshow(np.real(sig0), aspect='auto', interpolation='none')
    plt.figure()
    plt.plot(np.median(np.abs(sig0), axis=0))
    plt.show()
    exit()
    """
    
    skip = 0 # samples to skip from the beginning of the data
    n_samples = 40*10000 # no. of samples to keep (how big a dataset to work on)

    sig0 = sig0[skip:n_samples+skip]
    sig1 = sig1[skip:n_samples+skip]

    chans = read_4bit.read_header(f"{data_dir}{fname}")['chans']

    # pick out one orbcomm channel. acheived by looking. this one works for the lab data
    fi = int(7.531e6)
    ff = int(7.543e6)

    ipfb_chunk = 4*1000
    ifft_chunk = 2**12 # powers of 2 are best

    # trim to integer chunk count
    sig0 = sig0[:n_samples-n_samples%ifft_chunk]
    sig1 = sig1[:n_samples-n_samples%ifft_chunk]

    print('sig0 trimmed again')
    print(sig0.shape)

    print(f'number of chunks: {sig0.shape[0]//ifft_chunk}')
    
    # dimensions are (N of chunks, N of frequency bins)
    sig0_fft = np.zeros((sig0.shape[0]//ifft_chunk, ifft_chunk*2048+1), dtype=np.complex128)
    sig1_fft = np.zeros((sig0.shape[0]//ifft_chunk, ifft_chunk*2048+1), dtype=np.complex128)
    
    # work in chunks on the pfb'ed baseband, ipfb'ing each chunk and putting it in fourier space
    # that is sig0_fft[n] is the nth chunk as a function of frequency
    for i in range(sig0.shape[0]//ifft_chunk):
        print(f'iteration {i}')

        sig0_fft[i] = get_zerofilled_ffts(sig0[i*ifft_chunk:(i+1)*ifft_chunk], chans, N=ipfb_chunk)
        sig1_fft[i] = get_zerofilled_ffts(sig1[i*ifft_chunk:(i+1)*ifft_chunk], chans, N=ipfb_chunk)

    # gaussian window to pick out ORBCOMM channels:
    fwhm = ff - fi
    mu = (ff + fi)/2
    sig = fwhm / (2*np.sqrt(2*np.log(2)))
    xs = np.arange(sig0_fft.shape[1])
    gauss_window = 1/(sig*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sig**2))

    # apply window
    sig0_fft = sig0_fft * gauss_window
    sig1_fft = sig1_fft * gauss_window

    # save the result to be work on in another script (plot_timing.py)
    # this above process takes a while, even on modest sized data, thus the saving

    # I save (index, average of all pfb chunks (fct. of freq.), standard deviation of same)
    # std. dev. across chunks stands in for a better error estimate in the chi^2 calculation
    print(f"saving data at {name}_sig0")
    np.save(f"{name}_sig0", (xs, np.mean(sig0_fft, axis=0), np.std(sig0_fft, axis=0)))

    print(f"saving data at {name}_sig1")
    np.save(f"{name}_sig1", (xs, np.mean(sig1_fft, axis=0), np.std(sig1_fft, axis=0)))
    
    # the story continues in plot_timing.py
