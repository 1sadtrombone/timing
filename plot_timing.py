import numpy as np
import matplotlib.pyplot as plt

pol0 = np.load("data/lab_timestream_with_errors_97chunks_pol0.npy")
pol1 = np.load("data/lab_timestream_with_errors_97chunks_pol1.npy")
N = 97 # save this in future. HARDWIRED (this is number of pfb chunks. timing.py will tell you)

# limits of the orbcomm signal you're interested in
# again obtained only by looking
fi = int(7.531e6) 
ff = int(7.543e6)

# set up array of phase shifts to calculate the chi^2 of
phis_1d = np.linspace(-1e-2,1e-2,ff-fi)
fs, phis = np.meshgrid(np.arange(ff-fi)+fi, phis_1d)

# calculate the chi^2 of each phase shift (frequency is summed along so this is a function of phase shift only)
# note I apply a boxcar from fi to ff on top of the guass filt done in timing.py
# this is because I limit my frequency range to fi:ff, you could expand that

#chi2 = np.sum(2*np.imag(data[1,fi:ff]*np.exp(-1j*fs*phis))/np.abs(data[2,fi:ff])**2, axis=1)
chi2 = np.sum(np.abs(pol0[1,fi:ff]*np.exp(1j*fs*phis)-pol1[1,fi:ff])**2/np.abs(pol0[2,fi:ff]/np.sqrt(N))**2, axis=1)


print(np.sum(chi2/chi2.size))
print(chi2.size)
print(f"min: {phis_1d[np.where(chi2 == np.min(chi2))]}")

print("plotting..")
plt.plot(phis_1d, chi2, 'k')
plt.show()
