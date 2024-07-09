import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from hw3_helper_utils import create_motion_blur_filter
from wiener_filtering import *
def main():
    # Path to the image inside the subfolder - CHANGE IF NEEDED
    image_path = 'D:/Σχολή/8ο εξάμηνο/Ψηφιακή Επεξεργασία Εικόνας/hw3/'
    image_name = 'cameraman'
    
    # Image loading and preprocessing
    img = Image.open(image_path + image_name + '.tif')   
    if img == None:
            print("Error while loading the image")
            exit(1)
    x = np.array(img).astype(float)
    if np.any(x > 1.0):
        x *= 1/255

    # Parameters configuration
    noise_level = 0.02
    length = 20
    angle = 30

    # Zero Padding Handling 
    M, N = x.shape
    if length > M:
         x = add_zero_padding(x, int((length - M) / 2))
    if length > N:
         x = add_zero_padding(x, int((length - N) / 2))

    # Motion blur filter
    h = create_motion_blur_filter(length=length, angle=angle)
    y0 = convolve(x, h, mode="wrap")

    # White noise
    v = noise_level * np.random.randn(*x.shape)
    y = y0 + v        

    # Inverse filter
    x_inv = inverse_filtering(y, h)
    x_inv0 = inverse_filtering(y0, h)
    
    # K optimization
    K_values = np.linspace(1, 100, 1000) 
    best_K = find_best_K(y, h, K_values, image_path)
    print(f"Best K value found: {best_K:.2f}")

    # Optimal Wiener Filter
    x_hat = my_wiener_filter(y, h, best_K)

    # Zero Padding Handling
    if length > M:
         x = remove_zero_padding(x, int((length - M) / 2))
         y = remove_zero_padding(y, int((length - M) / 2))
         y0 = remove_zero_padding(y0, int((length - M) / 2))
         x_inv = remove_zero_padding(x_inv, int((length - M) / 2))
         x_inv0 = remove_zero_padding(x_inv0, int((length - M) / 2))
         x_hat = remove_zero_padding(x_hat, int((length - M) / 2))   
    if length > N:
         x = remove_zero_padding(x, int((length - N) / 2))
         y = remove_zero_padding(y, int((length - N) / 2))
         y0 = remove_zero_padding(y0, int((length - N) / 2))
         x_inv = remove_zero_padding(x_inv, int((length - N) / 2))
         x_inv0 = remove_zero_padding(x_inv0, int((length - N) / 2))
         x_hat = remove_zero_padding(x_hat, int((length - N) / 2))   
    
    # Result Visualization
    font_size = 10
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(f'Wiener Filter Application with parameters:\nlength={length}  angle={angle}  noise level={noise_level}  K={best_K:.2f}', fontsize=12)
    axs[0][0].imshow(x, cmap='gray')
    axs[0][0].set_title("Original image x", fontsize=font_size)
    axs[0][1].imshow(y0, cmap='gray')
    axs[0][1].set_title("Blurred image y0", fontsize=font_size)
    axs[0][2].imshow(y, cmap='gray')
    axs[0][2].set_title("Blurred and noisy image y", fontsize=font_size)
    axs[1][0].imshow(x_inv0, cmap='gray')
    axs[1][0].set_title("Inverse filtering\nnoiseless output x_inv0", fontsize=font_size)
    axs[1][1].imshow(x_inv, cmap='gray')
    axs[1][1].set_title("Inverse filtering\nnoisy output x_inv", fontsize=font_size)
    axs[1][2].imshow(x_hat, cmap='gray')
    axs[1][2].set_title("Wiener filtering\noutput x_hat", fontsize=font_size)
    for i in range(2):
         for j in range(3):
            axs[i][j].axis('off')
    fig.savefig(image_path + image_name + '_figures.png')

# Run the main function
main()