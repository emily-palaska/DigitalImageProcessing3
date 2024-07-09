import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Function to add zero padding
def add_zero_padding(image, pad_width):
    return np.pad(image, pad_width, mode='constant', constant_values=0)

# Function to remove zero padding
def remove_zero_padding(padded_image, pad_width):
    return padded_image[pad_width:-pad_width, pad_width:-pad_width]

def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    """
    Implements a Wiener filter for signal estimation.

    Args:
    - y: Observed signal (1D numpy array)
    - h: Impulse response of the filter (1D numpy array)
    - K: Power spectral density ratio (float)

    Returns:
    - Estimated signal (1D numpy array)
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

def j_curve(y: np.ndarray, h: np.ndarray, K_values: np.ndarray) -> np.ndarray:
    mse_values = []
    
    for K in K_values:
        # Compute the filtered signal using the Wiener filter
        x_hat = my_wiener_filter(y, h, K)
        
        # Compute Mean Squared Error (MSE) between the original signal y and filtered signal x_hat
        mse = np.mean((y - x_hat)**2)
        mse_values.append(mse)
    
    return np.array(mse_values)

def find_best_K(y: np.ndarray, h: np.ndarray, K_values: np.ndarray, image_path: str) -> float:
    # Compute MSE values for given K values
    mse_values = j_curve(y, h, K_values)
    
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
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, Y.shape)
    Y_inv = np.fft.ifft2(Y / (H + 10**(-10)))
    return np.abs(Y_inv)
    


