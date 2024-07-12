import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Function to add zero padding
def add_zero_padding(image:np.ndarray, pad_width:int) -> np.ndarray:
    """
    Adds zero padding to an image for filter convolution
    
    Args:
    - image: the numpy array representing the image to be padded
    - pad_width: the ammount of pizels that will be added to all 4 sides of the image

    Returns:
    - Padded image    
    """
    return np.pad(image, pad_width, mode='constant', constant_values=0)

# Function to remove zero padding
def remove_zero_padding(padded_image:np.ndarray, pad_width:int) -> np.ndarray:
    """
    Removes zero padding to an image afer filter convolution
    
    Args:
    - image: the padded image
    - pad_width: the ammount of pizels that will be added to all 4 sides of the image

    Returns:
    - Image without padding   
    """
    return padded_image[pad_width:-pad_width, pad_width:-pad_width]

def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    """
    Implements a Wiener filter for signal estimation.

    Args:
    - y: Observed signal (2D numpy array)
    - h: Impulse response of the filter (2D numpy array)
    - K: Power spectral density ratio (float)

    Returns:
    - Estimated signal (2D numpy array)
    """
    # Compute the 2D Fourier transform of the observed signal and the impulse response
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, Y.shape)
    
    # Compute the Wiener filter in the frequency domain
    H_conj = np.conj(H)
    H_abs_square = np.abs(H)**2
    G = H_conj / (H_abs_square + 1/K)
    
    # Apply the Wiener filter to the observed signal
    X_hat = G * Y
    
    # Compute the inverse Fourier transform to get the filtered signal
    x_hat = np.fft.ifft2(X_hat)
    
    # Return the real part of the filtered signal
    return np.real(x_hat)

def j_curve(x:np.ndarray, y: np.ndarray, h: np.ndarray, K_values: np.ndarray) -> np.ndarray:
    """
    Computes the error values of the implemented Wiener Filter for a set range of K values

    Args:
    - y: Observed signal (2D numpy array)
    - h: Impulse response of the filter (2D numpy array)
    - K_values: Range of K values

    Returns:
    - Mean Square Errors for every K values
    """
    mse_values = []
    
    for K in K_values:
        # Compute the filtered signal using the Wiener filter
        x_hat = my_wiener_filter(y, h, K)
        
        # Compute Mean Squared Error (MSE) between the original signal y and filtered signal x_hat
        mse = np.mean((x - x_hat)**2)
        mse_values.append(mse)
    
    return np.array(mse_values)

def find_best_K(x:np.ndarray, y: np.ndarray, h: np.ndarray, K_values: np.ndarray, image_path: str) -> float:
    """
    Finds the optimal K value of the Wiener Filter for MSE minimization
    Saves the J Curve graph in a designated path

    Args:
    - y: Observed signal (2D numpy array)
    - h: Impulse response of the filter (2D numpy array)
    - K_values: Range of K values
    - image_path: the path to save the graph

    Returns:
    - K value with minimum MSE
    """
    
    # Compute MSE values for given K values
    mse_values = j_curve(x, y, h, K_values)
    
    # Find the index of the minimum MSE
    best_K_index = np.argmin(mse_values)

    # Plot the J-curve (optional but useful for visualization)
    plt.figure()
    plt.plot(K_values, mse_values, color='#704c5e')
    plt.xlabel('K values')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('J-Curve for Wiener Filter')
    plt.plot(K_values[best_K_index], mse_values[best_K_index], marker='o', markersize=5, color='#558b6e', linestyle='None', label=f'Optimal K: {K_values[best_K_index]:.2f}')
    plt.grid(True)
    plt.legend()
    plt.savefig(image_path + "jcurve.png")
    
    # Return the best K value
    return K_values[best_K_index]

def inverse_filtering(y: np.ndarray, h:np.ndarray):
    """
    Applies the reverse filter to an image

    Args:
    - y: Observed signal (2D numpy array)
    - h: Impulse response of the filter to be reversed (2D numpy array)

    Returns:
    - Signal with reversed filter applied
    """
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, Y.shape)
    Y_inv = np.fft.ifft2(Y / (H + 10**(-10)))
    return np.abs(Y_inv)
    


