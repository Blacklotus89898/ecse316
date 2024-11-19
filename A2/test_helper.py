import numpy as np

import fft as our_implementations
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Test function to view the command call arguments
def test_command_call():
    mode, image = our_implementations.parse_command_call()
    print("Mode: ", mode)
    our_implementations.open_image(image)


# Test function for the DFT against the NumPy implementation
def test_DFT():
    test_cases = [
        [1, 2, 3, 4],
        [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j],
        [np.sin(2 * np.pi * 0.1 * i) for i in range(10)],
    ]

    for i in test_cases:
        our_X = our_implementations.DFT(i)
        NumPy_X = np.fft.fft(i)

        print("\nTesting DFT on: ", i)
        if np.allclose(our_X, NumPy_X, rtol=1e-15):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_X)
            print("NumPy implementation: ", NumPy_X)
    

# Test function for the IDFT against the NumPy implementation  
def test_IDFT():
    test_cases = [
        [1, 2, 3, 4],
        [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j],
        [np.sin(2 * np.pi * 0.1 * i) for i in range(10)],
    ]

    for i in test_cases:
        our_x = our_implementations.IDFT(i)
        NumPy_x = np.fft.ifft(i)

        print("\nTesting IDFT on: ", i)
        if np.allclose(our_x, NumPy_x, rtol=1e-15):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_x)
            print("NumPy implementation: ", NumPy_x)


# Test function for the 2D DFT against the NumPy implementation
def test_DFT2D():
    test_cases = [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j], [6 + 7j, 7 + 8j, 8 + 9j, 9 + 10j]],
        [[np.sin(2 * np.pi * 0.1 * i) for i in range(10)], [np.sin(2 * np.pi * 0.25 * i) for i in range(10)]],
    ]

    for i in test_cases:
        our_X = our_implementations.DFT2D(i)
        NumPy_X = np.fft.fft2(i)

        print("\nTesting DFT2D on: ", i)
        if np.allclose(our_X, NumPy_X, rtol=1e-8):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_X)
            print("NumPy implementation: ", NumPy_X)

# Test function for the FFT against the NumPy implementation
def test_FFT():
    test_cases = [
        [1, 2, 3, 4],
        [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j],
        [np.sin(2 * np.pi * 0.1 * i) for i in range(16)]
    ]

    for i in test_cases:
        our_X = our_implementations.FFT(i)
        NumPy_X = np.fft.fft(i)
        x = our_implementations.fft(i)

        print("\nTesting FFT on: ", i)
        if np.allclose(our_X, NumPy_X, rtol=1e-15):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_X)
            print("NumPy implementation: ", NumPy_X)

        # if np.array_equal(our_X, x):
        #     print("\033[32mTest passed\033[0m")
        # else:
        #     print("\033[31mTest failed\033[0m")
        #     print("Our implementation: ", our_X)
        #     print("Our x: ", x)


def test_IFFT():
    test_cases = [
        [1, 2, 3, 4],
        [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j],
        [np.sin(2 * np.pi * 0.1 * i) for i in range(16)]
    ]

    for i in test_cases:
        our_x = our_implementations.IFFT(i)
        NumPy_x = np.fft.ifft(i)
        x = our_implementations.inverse_fft(i)

        print("\nTesting IFFT on: ", i)
        if np.allclose(our_x, NumPy_x, rtol=1e-15):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_x)
            print("NumPy implementation: ", NumPy_x)

        # if np.array_equal(our_x, x):
        #     print("\033[32mTest passed\033[0m")
        # else:
        #     print("\033[31mTest failed\033[0m")
        #     print("Our implementation: ", our_x)
        #     print("Our x: ", x)
        

def test_FFT2D():
    test_cases = [
      [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j], [6 + 7j, 7 + 8j, 8 + 9j, 9 + 10j]],
        [[np.sin(2 * np.pi * 0.1 * i) for i in range(8)], [np.sin(2 * np.pi * 0.25 * i) for i in range(8)]],
        [[np.sin(2 * np.pi * 0.1 * i) for i in range(16)], [np.sin(2 * np.pi * 0.25 * i) for i in range(16)]],
    ]

    for i in test_cases:
        our_X = our_implementations.FFT2D(i)
        NumPy_X = np.fft.fft2(i)
        x = our_implementations.fft_2d(i)

        print("\nTesting FFT2D on: ", i)
        if np.allclose(our_X, NumPy_X, rtol=1e-15):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_X)
            print("NumPy implementation: ", NumPy_X)

        # if np.array_equal(our_X, x):
        #     print("\033[32mTest passed\033[0m")
        # else:
        #     print("\033[31mTest failed\033[0m")
        #     print("Our implementation: ", our_X)
        #     print("Our x: ", x)

def test_IFFT2D():
    test_cases = [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j], [6 + 7j, 7 + 8j, 8 + 9j, 9 + 10j]],
        [[np.sin(2 * np.pi * 0.1 * i) for i in range(8)], [np.sin(2 * np.pi * 0.25 * i) for i in range(8)]],
        [[np.sin(2 * np.pi * 0.1 * i) for i in range(16)], [np.sin(2 * np.pi * 0.25 * i) for i in range(16)]],
    ]

    for i in test_cases:
        our_x = our_implementations.IFFT2D(i)
        NumPy_x = np.fft.ifft2(i)
        x = our_implementations.inverse_fft_2d(i)

        print("\nTesting IFFT2D on: ", i)
        if np.allclose(our_x, NumPy_x, rtol=1e-15):
            print("\033[32mTest passed\033[0m")
        else:
            print("\033[31mTest failed\033[0m")
            print("Our implementation: ", our_x)
            print("NumPy implementation: ", NumPy_x)

        # if np.array_equal(our_x, x):
        #     print("\033[32mTest passed\033[0m")
        # else:
        #     print("\033[31mTest failed\033[0m")
        #     print("Our implementation: ", our_x)
        #     print("Our x: ", x)

def fast_mode_test(image):
    # Read the image using OpenCV
    image_array = cv.imread("moonlanding.jpg", cv.IMREAD_UNCHANGED)
    print(image_array.shape)

    # Convert to grayscale if the image has multiple channels
    if len(image_array.shape) == 3:
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    # Convert the image to a numpy array of floats
    image_array = np.asarray(image_array, dtype=complex)
    # transformed_image = FFT2D(image_array)
    transformed_image = np.fft.fft2(image_array)
    transformed_image = np.fft.ifft2(transformed_image)
    # tt = IFFT2D(transformed_image)
    
    # Plot the original and transformed images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(np.abs(image_array), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Transformed image
    axes[1].imshow(np.abs(transformed_image), cmap='gray', norm=colors.LogNorm())
    axes[1].set_title('2D FFT of the Image')
    axes[1].axis('off')
    
    plt.show()

def main():
    # test_command_call() 
    test_DFT()
    test_IDFT()
    test_DFT2D()

    # FFT tests ##############################################################
    #test_FFT()
    #test_IFFT()
    #test_FFT2D()
    #test_IFFT2D()
    # test_fft()
    # test_fft2D()

    # fast_mode_test("moonlanding.jpg")
    # fast_mode_test("moonlanding.jpg")

if __name__ == "__main__":
    main()