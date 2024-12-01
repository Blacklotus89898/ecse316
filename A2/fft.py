import argparse
import time
import cv2 as cv
import numpy as np
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
The formula provided in the assignment description is: 
    F_kl = Σ_{n=0}^{N-1} (Σ_{m=0}^{M-1} f_mn * e^(-2πikm/M)) * e^(-2πikn/N)) for k = 0, 1, ..., M-1 and l = 0, 1, ..., N-1
But it was only used as a reference to understand the concept of the 2D DFT. The actual 
implementation is different because matrices are structured as (rows, columns) but the 
signals in the formula are structured as (columns, rows).
'''
def DFT2D(signal):
    # Upon receiving a 2D signal, make sure it is a numpy tuple of vectors
    f = np.asarray(signal)  # If the signal is already a tuple of vectors, this will not change anything

    # 1D DFT of the rows
    rows = np.array([DFT(row) for row in f])

    # Transform the rows
    rows_transformed = rows.T

    # 1D DFT of the columns
    columns = np.array([DFT(col) for col in rows_transformed])

    # Transform the obtained array to match the structure of the 2D DFT
    F = columns.T

    return F





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


# padding the image to maintain the dimensions as a power of 2
def pad_image(image):
    """
    Pads the input image to the next power of 2 dimensions.
    """
    new_shape = [int(np.power(2, np.ceil(np.log2(dim)))) for dim in image.shape]
    padded_image = np.zeros(new_shape, dtype=image.dtype)
    padded_image[:image.shape[0], :image.shape[1]] = image

    return padded_image

'''
Mode 1: Convert image to FFT and plot
'''
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
    axes[1].imshow(np.abs(transformed_image[:image_array.shape[0], :image_array.shape[1]]), cmap='gray', norm=colors.LogNorm())
    axes[1].set_title('2D FFT of the Image')
    
    plt.show()


'''
Mode 2: Denoise image (FFT) and plot the image with truncated high frequencies
'''
def denoise_mode(image):
    # Read the image using OpenCV
    image_array = cv.imread(image, cv.IMREAD_GRAYSCALE)

    # Perform the 2D FFT on the padded image
    transformed_image = FFT2D(pad_image(image_array))
    # Zero out high frequency components based on a threshold
    
    non_zeros = np.count_nonzero(transformed_image)
    print("Original non-zero count: ", non_zeros)  
    
    # Calculate the magnitude of the transformed image
    magnitude = np.abs(transformed_image)
    
    # Determine the threshold for middle frequencies
    range = 0.1
    max = np.max(magnitude)

    # Zero out components within the middle frequency range
    transformed_image[(magnitude > max*range) & (magnitude < max*(1-range) )] = 0
    
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

    # Calculate the non-zero count of the denoised image
    filtered_non_zeros = np.count_nonzero(np.abs(transformed_image[:image_array.shape[0], :image_array.shape[1]]))
    print("Denoised non-zero count: ", filtered_non_zeros)
    print("Compression ratio: ", filtered_non_zeros / non_zeros)
    
    plt.show()


'''
Mode 3: Compress image and plot
'''
def compress_mode(image_path):
    """
    Compress an image using FFT by thresholding Fourier coefficients,
    and visualize the results at different compression levels.
    """
    # Load the image in grayscale
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found or invalid path.")
        return
    
    # Compression levels (percent of coefficients to retain)
    compression_levels = [0, 20, 40, 60, 80, 99.9]
    
    # Set up subplots
    _, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Display the original image in the first subplot
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image (0% Compression)")
    
    # Pad the image and perform FFT
    padded_image = pad_image(image)
    fft_transformed = FFT2D(padded_image)
    magnitude = np.abs(fft_transformed)
    original_non_zeros = np.count_nonzero(fft_transformed)
    print(f"Original non-zero coefficients: {original_non_zeros}")
    
    # Store original dimensions for cropping after inverse FFT
    original_shape = image.shape
    
    for i, level in enumerate(compression_levels):
        if level == 0:
            # Skip the original image; already displayed
            continue
        
        # Calculate thresholds for compression
        low_threshold = np.percentile(magnitude, level)  # Retain top 'level' percent of coefficients
        
        # Create a mask to retain only the largest coefficients
        compressed_fft = fft_transformed.copy()
        compressed_fft[magnitude < low_threshold] = 0  # Zero out smaller coefficients
        
        # Count non-zero coefficients after compression
        non_zero_count = np.count_nonzero(compressed_fft)
        print(f"Non-zero coefficients at {level}% compression: {non_zero_count}")
        
        # Perform inverse FFT to reconstruct the image
        compressed_image = IFFT2D(compressed_fft).real
        compressed_image = compressed_image[:original_shape[0], :original_shape[1]]  # Crop back to original size
        
        # Plot the compressed image
        row, col = divmod(i, 3)
        axes[row, col].imshow(compressed_image, cmap='gray')
        axes[row, col].set_title(f"{level}% Compression")
    
    # Adjust subplot layout and show results
    plt.show()



'''
Mode 4: Plot runtime graphs for DTF abd FFT
'''
def plot_runtime_mode():
    # Determine the number of trials to run
    num_trials = 10

    # Determine the range of sizes for the 2D arrays
    sizes = [2**i for i in range(5, 10)]

    # Lists to store the data for plotting
    x = []
    y_dtf = []
    y_fft = []
    std_dev_dft = []
    std_dev_fft = []

    # Measure the runtime for each size
    for size in sizes:
        print(f"===============Running for size {size}...")

        # Generate a random 2D array of the specified size
        signal = np.random.random((size, size)) # Will use floating point numbers (continuous) between 0 and 1 in a uniform distribution

        # Store the runtime for DFT and FFT at each trial
        dft_runtimes = []
        fft_runtimes = []

        # Must run 10 times for the average runtime
        for iter in range(num_trials):
            print(f"- Running trial {iter + 1}...")

            # Measure the runtime for DFT
            dft_start_time = time.time()
            DFT2D(signal)
            dft_end_time = time.time()
            dft_runtimes.append(dft_end_time - dft_start_time)

            # Measure the runtime for FFT
            fft_start_time = time.time()
            FFT2D(signal)
            fft_end_time = time.time()
            fft_runtimes.append(fft_end_time - fft_start_time)

        # Calculate the average runtime for DFT and FFT
        mean_dft_runtime = np.mean(dft_runtimes)
        mean_fft_runtime = np.mean(fft_runtimes)
        x.append(size)
        y_dtf.append(mean_dft_runtime)
        y_fft.append(mean_fft_runtime)

        # Calculate the standard deviation for the runtimes with a confidence interval of 97%
        std_dev_dft_runtime = np.std(dft_runtimes)
        std_dev_fft_runtime = np.std(fft_runtimes)
        std_dev_dft.append(2 * std_dev_dft_runtime)
        std_dev_fft.append(2 * std_dev_fft_runtime)

        # Print the average runtime for DFT and FFT
        print(f"Average runtime for DFT: {mean_dft_runtime}")
        print(f"Average runtime for FFT: {mean_fft_runtime}")

        # Print the variance for the runtimes 
        print(f"Variance for DFT: {np.var(dft_runtimes)}")
        print(f"Variance for FFT: {np.var(fft_runtimes)}")

        # Print the standard deviation for the runtimes
        print(f"Standard deviation for DFT with a 97% confidence interval: {2 * std_dev_dft_runtime}")
        print(f"Standard deviation for FFT with a 97% confidence interval: {2 * std_dev_fft_runtime}") 

        print(f"Done running size {size}================")
        
    print("Now plotting...")

    # Set the plot labels
    plt.title('Runtime Comparison of 2D DFT and 2D FFT')
    plt.xlabel('Size of 2D Array')
    plt.ylabel('Runtime (s)')

    # Plot the runtime graph for DFT and FFT
    plt.xscale('log')
    plt.yscale('log')
    plt.errorbar(x, y_dtf, yerr=std_dev_dft, capsize=3, label="DFT", color='blue')
    plt.errorbar(x, y_fft, yerr=std_dev_fft, capsize=3, label="FFT", color='red')

    # Display the plot
    plt.legend()
    plt.savefig('2D_runtime_comparison.png', dpi=300)  # Save the plot as a PNG file with a high resolution
    plt.show()


def main():
    mode, image = parse_command_call()

    # Validate the image path
    if not cv.haveImageReader(image):
        print("Invalid image path. Please provide a valid image path.")
        return

    # Choose the appropriate functionality based on the mode
    if mode == 1:
        fast_mode(image)
    elif mode == 2:
        denoise_mode(image)
    elif mode == 3:
        compress_mode(image)
    elif mode == 4:
        plot_runtime_mode()
    else:
        print("Invalid mode. Please choose a mode from 1 to 4.")


if __name__ == "__main__":
    main()
