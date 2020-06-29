import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "C:\\Users\\Akashbhardwaj\\Desktop\\Starboy.wav"

signal, sr = librosa.load(file, sr=22050)

librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
#plt.show()

fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
left_frequnecy = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]
plt.plot(left_frequnecy, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()