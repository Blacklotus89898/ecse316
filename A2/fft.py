import argparse
import cv2 as cv
import numpy as np
import plotly as ply
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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


# def FFT(signal):
#     """
#     A recursive implementation of the 1D Cooley-Tukey FFT.
#     The input should have a length that is a power of 2.
#     """
#     # Convert the input signal to a numpy array of complex numbers
#     x = np.asarray(signal, dtype=complex)
#     N = x.shape[0]

#     # Pad the array with zeros if the length is not a power of 2
#     if not np.log2(N).is_integer():
#         next_pow2 = int(np.power(2, np.ceil(np.log2(N))))
#         x = np.pad(x, (0, next_pow2 - N), mode='constant')
#         N = x.shape[0]

#     # Base case: if the input contains only one element, return it
#     if N <= 1:
#         return x

#     # Recursive case: split the array into even and odd parts
#     even = FFT(x[0::2])
#     odd = FFT(x[1::2])

#     # Compute the twiddle factors
#     T = np.exp(-2j * np.pi * np.arange(N) / N)[:N // 2]

#     # Combine the results of the even and odd parts
#     result = np.concatenate([even + T * odd, even - T * odd])

#     return result
def FFT(signal):
    """
    A recursive implementation of the 1D Cooley-Tukey FFT.
    The input should have a length that is a power of 2.
    """
    x = np.asarray(signal, dtype=complex)
    original_N = x.shape[0]
    N = x.shape[0]

    # Pad the array with zeros if the length is not a power of 2
    if not np.log2(N).is_integer():
        next_pow2 = int(np.power(2, np.ceil(np.log2(N))))
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = x.shape[0]

    if N <= 1:
        return x

    # Recursive case: split the array into even and odd parts
    even = FFT(x[0::2])
    odd = FFT(x[1::2])

    # Compute the twiddle factors
    T = np.exp(-2j * np.pi * np.arange(N) / N)[:N // 2]

    # Combine the results of the even and odd parts
    result = np.concatenate([even + T * odd, even - T * odd])

    # Remove the padding before returning the result
    return result[:original_N]

# #The Cooley-Tukey FFT
# def fft(signal):
#     N = len(signal)

#     if N <= 1:
#         return signal

#     even = fft(signal[0::2])
#     odd = fft(signal[1::2])

#     t = np.exp(-2j * np.pi * np.arange(N) / N)
#     return np.concatenate([even + t[:N//2] * odd, even + t[N//2:] * odd])

# def fft_2D(matrix: np.ndarray):
#     assert type(matrix) is np.ndarray

#     res1 = np.zeros(matrix.shape, dtype='complex_')

#     for i, row in enumerate(matrix):
#         # if DEBUG:
#         #     if i % 100 == 0:
#         #         print(i)
#         res1[i] = fft(row)

#     res2 = np.zeros(matrix.T.shape, dtype='complex_')
#     for i, col in enumerate(res1.T):
#         # if DEBUG:
#         #     if i % 100 == 0:
#         #         print(i)
#         res2[i] = fft(col)

#     return res2.T

def IFFT(signal):
    """
    A recursive implementation of the 1D Cooley-Tukey IFFT.
    The input should have a length that is a power of 2.
    """
    # Helper function to perform the IFFT
    def _IFFT(x):
        N = x.shape[0]

        # Base case: if the input contains only one element, return it
        if N <= 1:
            return x

        # Recursive case: split the array into even and odd parts
        even = _IFFT(x[0::2])
        odd = _IFFT(x[1::2])

        # Compute the twiddle factors with a positive sign in the exponent
        T = np.exp(2j * np.pi * np.arange(N) / N)[:N // 2]

        # Combine the results of the even and odd parts
        result = np.concatenate([even + T * odd, even - T * odd])

        return result

    x = np.asarray(signal, dtype=complex)
    N = x.shape[0]

    # Pad the array with zeros if the length is not a power of 2
    if not np.log2(N).is_integer():
        next_pow2 = int(np.power(2, np.ceil(np.log2(N))))
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = x.shape[0]

    # Perform the IFFT using the helper function
    result = _IFFT(x)

    # Scale the result by the inverse of the length
    return result / N


# def FFT2D(signal):
#     """
#     A recursive implementation of the 2D Cooley-Tukey FFT.
#     The input should have dimensions that are powers of 2.
#     """
#     # Convert the input signal to a numpy array of complex numbers
#     f = np.asarray(signal, dtype=complex)
#     om, on = f.shape
#     N, M = f.shape

#     # Pad the array with zeros if the dimensions are not powers of 2
#     if not (np.log2(N).is_integer() and np.log2(M).is_integer()):
#         next_pow2_N = int(np.power(2, np.ceil(np.log2(N))))
#         next_pow2_M = int(np.power(2, np.ceil(np.log2(M))))
#         f = np.pad(f, ((0, next_pow2_N - N), (0, next_pow2_M - M)), mode='constant')
#         N, M = f.shape

#     # Perform the 1D FFT on the rows
#     F = np.array([FFT(row) for row in f])

#     # Perform the 1D FFT on the columns
#     F = np.array([FFT(col) for col in F.T]).T

#     # Remove the padding if it was added
#     F = F[:om, :on]

#     return F


def FFT2D(signal):
    """
    A 2D FFT implementation using the 1D FFT function.
    """
    signal = np.asarray(signal, dtype=complex)
    original_shape = signal.shape
    N, M = signal.shape[:2]

    # Pad the array with zeros if the dimensions are not powers of 2
    if not np.log2(N).is_integer():
        next_pow2_N = int(np.power(2, np.ceil(np.log2(N))))
        pad_width = ((0, next_pow2_N - N), (0, 0)) + ((0, 0),) * (signal.ndim - 2)
        signal = np.pad(signal, pad_width, mode='constant')
        N = signal.shape[0]
    if not np.log2(M).is_integer():
        next_pow2_M = int(np.power(2, np.ceil(np.log2(M))))
        pad_width = ((0, 0), (0, next_pow2_M - M)) + ((0, 0),) * (signal.ndim - 2)
        signal = np.pad(signal, pad_width, mode='constant')
        M = signal.shape[1]

    # Apply FFT to each row
    F = np.zeros(signal.shape, dtype=complex)
    for i in range(N):
        F[i, :] = FFT(signal[i, :])

    # Apply FFT to each column
    for j in range(M):
        F[:, j] = FFT(F[:, j])

    # Remove the padding if it was added
    F = F[:original_shape[0], :original_shape[1]]

    return F

def IFFT2D(signal):
    """
    A recursive implementation of the 2D Cooley-Tukey IFFT.
    The input should have dimensions that are powers of 2.
    """

    # Convert the input signal to a numpy array of complex numbers
    F = np.array([np.array(dimension, dtype=complex) for dimension in signal])
    # F = np.array(signal, dtype=complex)
    om, on = F.shape
    N, M = F.shape

    # Pad the array with zeros if the dimensions are not powers of 2
    if not (np.log2(N).is_integer() and np.log2(M).is_integer()):
        next_pow2_N = int(np.power(2, np.ceil(np.log2(N))))
        next_pow2_M = int(np.power(2, np.ceil(np.log2(M))))
        F = np.pad(F, ((0, next_pow2_N - N), (0, next_pow2_M - M)), mode='constant')
        N, M = F.shape

    # Perform the 1D IFFT on the rows
    F = np.array([IFFT(row) for row in F])

    # Perform the 1D IFFT on the columns
    F = np.array([IFFT(col) for col in F.T]).T

    # Scale the result by the inverse of the dimensions
    return F[:om, :on]

# def dft(signal):
#     signal = np.asarray(signal, dtype=complex) # Put input as array with complex type
#     N = signal.shape[0]   # Return number of rows in array

#     # Instantiate array of zeros of length rows with complex type
#     result = np.zeros(N, dtype=complex)

#     for k in range(N):
#         for n in range(N):
#             # DFT formula
#             result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

#     return result

# def inverse_dft(X):
#     X = np.asarray(X, dtype=complex)
#     N = X.shape[0]
#     inverse = np.zeros(N, dtype=complex)

#     for n in range(N):
#         for k in range(N):
#             inverse[n] += X[k] * np.exp(2j * np.pi * k * n / N)

#         inverse[n] = inverse[n] / N

#     return inverse

# def dft_2d(signal):
#     signal = np.asarray(signal, dtype=complex)

#     # Store both dimensions of the array
#     N = signal.shape[0]
#     M = signal.shape[1]

#     # Instantate arrays of 0s with same dimensions 
#     row = np.zeros((N, M), dtype=complex) 
#     result = np.zeros((M, N), dtype=complex) 
    
#     # Compute the dft of input and store it in row
#     for n in range(N):
#         row[n] = dft(signal[n])

#     column = row.transpose()

#     # Compute dft of transposed array
#     for m in range(M):
#         result[m] = dft(column[m])

#     return result.transpose()

# def fft(signal):
#     threshold = 16
#     signal = np.asarray(signal, dtype=complex)
#     N = signal.shape[0]

#     if N > threshold:
#         even = fft(signal[::2])
#         odd = fft(signal[1::2])
#         result = np.zeros(N, dtype=complex)
#         for n in range(N):
#             result[n] = even[n % (N // 2)] + np.exp(-2j * np.pi * n / N) * odd[n % (N // 2)]
#         return result
#     else:
#         return dft(signal)

# def inverse_fft(signal):
#     threshold = 16
#     signal = np.asarray(signal, dtype=complex)
#     N = signal.shape[0]

#     if N > threshold:
#         even = inverse_fft(signal[::2])
#         odd = inverse_fft(signal[1::2])
#         result = np.zeros(N, dtype=complex)
#         for n in range(N):
#             result[n] = (N // 2) * even[n % (N // 2)] + np.exp(2j * np.pi * n / N) * (N // 2) * odd[n % (N // 2)]
#             result[n] = result[n] / N

#         return result
#     else:
#         return inverse_dft(signal)

# def fft_2d(signal):
#     signal = np.asarray(signal, dtype=complex)
#     width, height = signal.shape
#     result = np.zeros((width, height), dtype=complex)

#     for i in range(height):
#         result[:, i] = fft(signal[:, i])
#     for i in range(width):
#         result[i, :] = fft(result[i, :])

#     return result

# def inverse_fft_2d(signal):
#     signal = np.asarray(signal, dtype=complex)
#     width, height = signal.shape
#     result = np.zeros((width, height), dtype=complex)

#     for i in range(width):
#         result[i, :] = inverse_fft(signal[i, :])
#     for i in range(height):
#         result[:, i] = inverse_fft(result[:, i])

#     return result

# padding the image to maintain the dimensions as a power of 2
def pad_image(image):
    """
    Pads the input image to the next power of 2 dimensions.
    """
    new_shape = [int(np.power(2, np.ceil(np.log2(dim)))) for dim in image.shape]
    padded_image = np.zeros(new_shape, dtype=image.dtype)
    padded_image[:image.shape[0], :image.shape[1]] = image

    return padded_image

def fast_mode(image):
    # Read the image using OpenCV
    image_array = cv.imread(image, cv.IMREAD_UNCHANGED)

    # Convert to grayscale if the image has multiple channels
    if len(image_array.shape) == 3:
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    
    # Convert the image to a numpy array of floats
    image_array = np.asarray(image_array, dtype=complex)
    
    # Perform the 2D FFT on the padded image
    transformed_image = FFT2D(pad_image(image_array))
    
    # Plot the original and transformed images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(np.abs(image_array), cmap='gray')
    axes[0].set_title('Original Image')
    
    # Transformed image
    axes[1].imshow(np.abs(transformed_image), cmap='gray', norm=colors.LogNorm())
    axes[1].set_title('2D FFT of the Image')
    
    plt.show()

def denoise_mode(image):
    # Read the image using OpenCV
    image_array = cv.imread(image, cv.IMREAD_GRAYSCALE)

    # Perform the 2D FFT on the padded image
    transformed_image = FFT2D(pad_image(image_array))

    # Zero out high frequency components based on a threshold
    threshold = np.max(np.abs(transformed_image)) * 0.16
    transformed_image[np.abs(transformed_image) > threshold] = 0

    # Perform the 2D IFFT
    denoised_image = IFFT2D(transformed_image).real

    # Plot the original and denoised images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(np.abs(image_array), cmap='gray')
    axes[0].set_title('Original Image')

    # Denoised image
    axes[1].imshow(np.abs(denoised_image[:image_array.shape[0], :image_array.shape[1]]), cmap='gray')
    axes[1].set_title('Denoised Image')

    plt.show()

def compress_mode(image):
    image_array = cv.imread(image, cv.IMREAD_GRAYSCALE)

    # Perform the 2D FFT on the padded image
    transformed_image = FFT2D(pad_image(image_array))

    # Filter the transformed image by keeping only the top and bottom values
    magnitude = np.abs(transformed_image)
    threshold = np.percentile(magnitude, 99.9)  # Keep top 5% values
    filtered_image = np.where(magnitude < threshold, transformed_image, 0)

    # Perform the 2D IFFT on the filtered image
    compressed_image = IFFT2D(filtered_image).real

    # Plot the original and compressed images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(np.abs(image_array), cmap='gray')
    axes[0].set_title('Original Image')

    # Compressed image
    axes[1].imshow(np.abs(compressed_image[:image_array.shape[0], :image_array.shape[1]]), cmap='gray')
    axes[1].set_title('Compressed Image')

    plt.show()
  
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
