import numpy as np

import fft as our_implementations

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


def main():
    #test_command_call() 
    test_DFT()
    test_IDFT()
    test_DFT2D()


if __name__ == "__main__":
    main()