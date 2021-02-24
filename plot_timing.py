import numpy as np
import matplotlib.pyplot as plt

data_file = "xcorr_lab_fspace.npy"

corr = np.load(f"data/{data_file}")
print(corr.shape)
N = corr.size
dt = 1/(250e6)
ts = np.linspace(0,N*dt,N) - N/2*dt

th = np.arctan2(np.imag(corr), np.real(corr))

plt.plot(th)
plt.show()
