# Ecse316 Fourier Transform Program


## Warning

Please note that our comments or instructions might directly use content from the document "ECSE316_A2_F2024.pdf" provided to us. 


## Install Dependencies

We incorporated a `requirements.txt` file to install the dependencies that our program uses. (Inspired by https://www.freecodecamp.org/news/python-requirementstxt-explained/)

Python version used: 3.9.6+

Please run the following line before running the program in the terminal. Make sure that you are in the directory of this project: 

```
pip install -r requirements.txt
```


## Command Line Arguments

-m mode (optional):
-   1: Fast mode: Convert image to FFT form and display. The default value.
-   2: Denoise: The image is denoised by applying an FFT, truncating high frequencies and then displayed
-   3: Compress: Compress image and plot.
-   4: Plot runtime graphs for the report.

-i image (optional): 
-   Filename of the image for the DFT. The default value is moonlanding.jpg, which is the image that was provided to us.


Please run the following command to run our program in the terminal. Make sure that you are in the directory of this project:

``` 
python ./fft.py [-m mode#] [-i image_path]
```


