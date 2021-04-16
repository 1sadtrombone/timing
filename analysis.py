import numpy as np
import matplotlib.pyplot as plt

#data_file = "data/lab_timestream_with_errors_97chunks.npy"
#data = np.load(data_file)

pol0 = np.load("data/lab_timestream_with_errors_97chunks_pol0.npy")
pol1 = np.load("data/lab_timestream_with_errors_97chunks_pol1.npy")
N = 97 # save this in future. HARDWIRED
"""
plt.plot(data[1])
plt.plot(data[2])
plt.show()
"""
fi = int(7.531e6)
ff = int(7.543e6)

phis_1d = np.linspace(-1e-2,1e-2,10000)
fs, phis = np.meshgrid(np.arange(ff-fi)+fi, phis_1d)

#chi2 = np.sum(2*np.imag(data[1,fi:ff]*np.exp(-1j*fs*phis))/np.abs(data[2,fi:ff])**2, axis=1)
chi2 = np.sum(np.abs(pol0[1,fi:ff]*np.exp(1j*fs*phis)-pol1[1,fi:ff])**2/np.abs(pol0[2,fi:ff]/np.sqrt(N))**2, axis=1)


print(np.sum(chi2/chi2.size))
print(chi2.size)
print(f"min: {phis_1d[np.where(chi2 == np.min(chi2))]}")

print("plotting..")
plt.plot(phis_1d, chi2, 'k')
plt.show()


"""
FOR PLOT IN PRESENTATION

data_file = "pol0_lab_timestream.npy"
err_file = "pol0_lab_fspace.npy"
pol0_file = "pol0_lab_fspace_gausswin.npy"
pol1_file = "pol1_lab_fspace_gausswin.npy"

initial_freq = 1830/2048 * 125
final_freq = 1855/2048 * 125

pol1 = np.load(f"data/{pol1_file}")
fs = np.linspace(initial_freq,final_freq,pol1.size)

plt.plot(fs, pol1, "k")
plt.xlabel("Frequency (MHz)", fontsize=20)
plt.ylabel("Signal (arb. units)", fontsize=20)
plt.title("Spectrum, East-West Polarization", fontsize=30)
plt.xlim([0.06+1.13e2, 0.07+1.13e2])
plt.xticks(np.arange(*[0.06+1.13e2, 0.07+1.13e2], 0.001))
ax=plt.gca()
ax.tick_params(axis='both', labelsize=15)
plt.show()

ts = np.load(f"data/{data_file}")
dt = 4e-9
time = np.arange(0, ts.size*dt, dt)
"""

"""
pol0 = np.fft.rfft(ts)
plt.plot(fs, pol0[1:], "k")
plt.xlabel("Frequency (MHz)", fontsize=20)
plt.ylabel("Signal (arb. units)", fontsize=20)
plt.title("Spectrum, North-South Polarization", fontsize=40)
ax=plt.gca()
ax.tick_params(axis='both', labelsize=15)
plt.show()
"""
"""
plt.plot(time, ts, "k")
plt.xlabel("Observing time (s)", fontsize=20)
plt.ylabel("Signal (arb. units)", fontsize=20)
plt.title("Timestream Data, North-South Polarization", fontsize=40)
ax=plt.gca()
ax.tick_params(axis='both', labelsize=15)
plt.show()
"""
"""
fi = int(7.531e6)
ff = int(7.543e6)

pol0 = np.load(f"data/{pol0_file}")[fi:ff]
pol1 = np.load(f"data/{pol1_file}")[fi:ff]

err_data = np.load(f"data/{err_file}")
#err2 = np.abs(np.mean(err_data[1000:fi//2]**2))
err2 = 1

# one f bin is one Hz, right?
phis_1d = np.linspace(-1e-2,1e-2,10000)
fs, phis = np.meshgrid(np.arange(ff-fi), phis_1d)

chi2 = np.sum(np.imag(pol0*np.exp(1j*fs*phis)-pol1)**2/err2, axis=1)

print(f"min: {phis_1d[np.where(chi2 == np.min(chi2))]}")

print("plotting...")
plt.plot(phis_1d*1000, chi2, "k")
plt.xlabel("Phase shift $\Delta t_\mathrm{clk}$ (ms)")
plt.ylabel("$\chi^2$")
plt.title("Finding Time Delay Between Signals")
plt.xlim([-2.5,2.5])
plt.margins(x=None)
ax = plt.gca()
#ax.tick_params(axis='both', labelsize=10)
plt.show()
"""
"""
corr = np.load(f"data/{data_file}")
err_corr = np.load(f"data/{err_file}")
print(corr.shape)
N = corr.size
dt = 1/(250e6)
ts = np.linspace(0,N*dt,N) - N/2*dt

#realspace = np.fft.irfft(corr)

fi = int(7.531e6)
ff = int(7.543e6)

err = np.sqrt(np.mean(err_corr[1000:fi//2]**2))
del err_corr

th = np.arctan2(np.imag(corr), np.real(corr))

print(np.mean(th[fi:ff]))

# one f bin is one Hz, right?
phis_1d = np.linspace(-1e-2,1e-2,10000)
fs, phis = np.meshgrid(np.arange(ff-fi), phis_1d)
print(fs.shape)

#chi2 = np.sum((np.abs(corr[fi:ff])*np.exp(1j*fs*phis) - corr[fi:ff])**2/np.abs(err)**2, axis=1)
chi2 = np.sum(np.imag(corr[fi:ff]*np.exp(1j*fs*phis))**2/np.abs(err)**2, axis=1)

print(chi2.shape)


print("plotting...")
plt.plot(phis_1d, chi2)
plt.show()
"""
