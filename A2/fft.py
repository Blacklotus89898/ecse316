import argparse
import cv2 as cv
import numpy as np
import plotly as ply


# Helper function to manage the arguments included in the command line call
# (inspired by https://docs.python.org/3/library/argparse.html)
def parse_command_call():
    parser = argparse.ArgumentParser()  
    parser.add_argument("-m", default=1, dest="mode", required=False, type=int)
    parser.add_argument("-i", default="moonlanding.jpg", dest="image", required=False, type=str)

    args = parser.parse_args()

    return args.mode, args.image


# Helper function to view an image through a window
# (inspired by: https://opencv.org/get-started/)
def open_image(image):
    img = cv.imread(image)
    cv.imshow(image, img)
    cv.waitKey(0)   # Must press any key to close the image window


'''
Helper function to perform the (naïve approach) Discrete Fourier Transform (DFT) on a signal.
Using the formula provided in the assignment description: X_k = Σ_{n=0}^{N-1} x_n * e^(-i2πkn/N) for k = 0, 1, ..., N-1
Let: 
   - X_k be the DFT of the signal at frequency k --> Implemented as a vector X[k]
   - n be the discrete time index
   - N be the number of samples taken from the signal
   - x_n be the signal at discrete time n --> Implemented as a vector x[n]
   - k be the frequency index

(DFT implementation inspired by: https://medium.com/@positive.delta.hm/implementing-the-discrete-fourier-transform-in-python-978dedded5bc)
(Vector handling inpired by NumPy's API: https://numpy.org/doc/stable/reference/routines.array-creation.html)
'''
def DFT(signal):
    # Upon receiving a signal, we must make sure it is a numpy array
    x = np.asarray(signal)  # If the signal is already a numpy array, this will not change anything 

    # Then, we set the number of samples taken from the signal
    N = len(x)

    # We initialize the DFT vector X[k] to contain all zeros but that can handle complex numbers 
    X = np.zeros(N, dtype=complex)

    # We can now calculate the DFT of the signal by performing the summation 
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N) 

    return X


'''
Helper function to perform the (naïve approach) Inverse Discrete Fourier Transform (IDFT) on a signal.
Using the formula provided in the assignment description: x_n = (1/N) * Σ_{k=0}^{N-1} X_k * e^(i2πkn/N) for n = 0, 1, ..., N-1
Let: 
   - x_n be the IDFT of the signal at discrete time n --> Implemented as a vector x[n]
   - N be the number of samples taken from the signal
   - X_k be the signal at frequency k --> Implemented as a vector X[k]
   - k be the frequency index
   - n be the discrete time index
(Vector handling inpired by NumPy's API: https://numpy.org/doc/stable/reference/routines.array-creation.html)
'''
def IDFT(signal):
    # Upon receiving a signal, we must make sure it is a numpy array
    X = np.asarray(signal)  # If the signal is already a numpy array, this will not change anything 

    # Then, we set the number of samples taken from the signal
    N = len(X)

    # We initialize the IDFT vector x[n] to contain all zeros but that can handle complex numbers 
    x = np.zeros(N, dtype=complex)

    # We can now calculate the IDFT of the signal by performing the summation 
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N) 

    # We must scale the IDFT vector by 1/N after the summation
    x = (1/N) * x

    return x


'''
Helper function to perform the (naïve approach) 2D Discrete Fourier Transform (DFT).
Using the formula provided in the assignment description: F_kl = Σ_{n=0}^{N-1} (Σ_{m=0}^{M-1} f_mn * e^(-2πikm/M)) * e^(-2πikn/N)) for k = 0, 1, ..., M-1 and l = 0, 1, ..., N-1
Let:
   - F_kl be the 2D DFT of the signal at frequency (k, l) --> Implemented as a matrix F[k, l]
   - f_mn be the signal at discrete time m and n --> Implemented as a matrix f[m, n]
   - N be the number of samples taken from the signal in the x-axis --> Number of rows
   - M be the number of samples taken from the signal in the y-axis --> Number of columns
   - k be the frequency index in the x-axis
   - l be the frequency index in the y-axis
   - n be the discrete time index in the y-axis
   - m be the discrete time index in the x-axis

(2D DFT understanding pulled from: https://www.corsi.univr.it/documenti/OccorrenzaIns/matdid/matdid027832.pdf)
'''
def DFT2D(signal):
    # Upon receiving a 2D signal, we must make sure it is a numpy tuple of vectors
    f = np.asarray(signal)  # If the signal is already a tuple of vectors, this will not change anything

    # Then, we set the number of samples taken from the signal
    N = f.shape[0]  # Number of rows
    M = f.shape[1]  # Number of columns

    # We initialize the DFT matrix F[k, l] to contain all zeros but that can handle complex numbers 
    F = np.zeros((N, M), dtype=complex)

    # We can now calculate the 2D DFT of the image by performing the summation 
    for k in range(N): 
        for l in range(M):  
            for m in range(N): 
                for n in range(M):  
                    F[k, l] += f[m, n] * np.exp(-2j * np.pi * k * m / N) * np.exp(-2j * np.pi * l * n / M)

    return F



def fast_mode(image):
    pass

def denoise_mode(image):
    pass

def compress_mode(image):
    pass

def plot_runtime_mode(image):
    pass









# (Opening the image inspired by: https://opencv.org/get-started/)
def main():
    mode, image = parse_command_call()

    # Choose the appropriate functionality based on the mode
    if mode == 1:
        fast_mode(image)
    elif mode == 2:
        denoise_mode(image)
    elif mode == 3:
        compress_mode(image)
    elif mode == 4:
        plot_runtime_mode(image)




if __name__ == "__main__":
    main()
