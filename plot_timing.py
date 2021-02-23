import numpy as np
import matplotlib.pyplot as plt

data_file = "xcorr_lab_gauss_window_with_packets.npy"

corr = np.load(f"data/{data_file}")
print(corr.shape)
N = corr.size
dt = 1/(250e6)
ts = np.linspace(0,N*dt,N) - N/2*dt

corr = np.hstack((corr[N//2:], corr[:N//2]))
corr2 = corr
corrft = np.fft.rfft(corr)
corrft2 = np.fft.rfft(corr2)
crud = np.conj(corrft)*corrft2
th=np.arctan2(np.imag(crud),np.real(crud))

plt.plot(th)
plt.show()
