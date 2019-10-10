import numpy as np
import os
from common import *
from matplotlib import pyplot as plt
import cv2
import math
from skimage.util import img_as_float
import copy
## Image Patches ##
def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N 
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function

    x = np.floor((image.shape[0])/16)
    y = np.floor((image.shape[1])/16)
    output = []
  
    for x1 in range(0,int(x)):
        for y1 in range(0, int(y)):
            each_patch = image[16*x1 : 16*(x1+1), 16*y1 : 16*(y1+1)]
            each_patch = each_patch / np.linalg.norm(each_patch)
            #display_img(each_patch)
            #save_img(each_patch, "./image_patches/q1_patch1.png" )
            output.append(each_patch)
   
    return output


## Gaussian Filter ##
def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W
    # padding with zero 
    print(math.e)
    image = np.pad(image, ((1, 1), (1, 1)), 'constant')
    output = copy.deepcopy(image)
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            output[x][y] = image[x - 1][y]*kernel[0] + image[x][y]*kernel[1] + image[x + 1][y]*kernel[2]
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            output[x][y] = image[x][y - 1]*kernel[0] + image[x][y]*kernel[1] + image[x][y + 1]*kernel[2]
    output = (output-np.min(output))/(np.max(output)-np.min(output))*255
    output = output.astype(np.uint8)
    return output

def convolve_for_axis(image, kernel, axis = 0):
    image = np.pad(image, ((1, 1), (1, 1)), 'constant')
    output = copy.deepcopy(image)
    # for x 
    if axis == 0:
        for x in range(1, image.shape[0] - 1):
            for y in range(1, image.shape[1] - 1):
                output[x][y] = image[x - 1][y]*kernel[0] + image[x][y]*kernel[1] + image[x + 1][y]*kernel[2]
    else:    
        for x in range(1, image.shape[0] - 1):
            for y in range(1, image.shape[1] - 1):
                output[x][y] = image[x][y - 1]*kernel[0] + image[x][y]*kernel[1] + image[x][y + 1]*kernel[2]
   # display_img(image)
    display_img(output)
    return output

## Edge Detection ##
def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Fix kx, ky
    kx = [-1/2, 0, 1/2]  # 1 x 3
    ky = [-1/2, 0, 1/2]  # 3 x 1

    Ix = convolve_for_axis(image, kx, 0)
    Iy = convolve_for_axis(image, ky, 1)
    # TODO: Use Ix, Iy to calculate grad_magnitude
    x = Ix**2 + Iy**2
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)
    normalized_img = (grad_magnitude-np.min(grad_magnitude))/(np.max(grad_magnitude)-np.min(grad_magnitude))*255
    grad_magnitude = normalized_img.astype(np.uint8)

    return grad_magnitude, Ix, Iy


## Sobel Operator ##
def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    Gx, Gy, grad_magnitude = None, None, None

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W
    # You are encouraged not to use sobel_operator() in this function.

    # TODO: Use convolve() to complete the function
    output = []

    return output




def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # Q1
    patches = image_patches(img)
    
    chosen_patches = patches[400]
    cv2.imwrite("./image_patches/q1_patch1.png", chosen_patches)
    
    # Q2: No code

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.  There is tolerance for the kernel.
    gau = (math.sqrt(math.log(2)/math.pi)*(math.pow(math.e, -math.log(2))))
   # x = np.dot(gau, np.transpose(gau))
    gaussian = np.array([gau, math.sqrt(math.log(2)/math.pi), gau])
    x = np.dot(gaussian, np.transpose(gaussian))
    kernel_gaussian = gaussian
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussia.png")
    
    # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    ########################
  
    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################

    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code

    print("LoG Filter is done. ")
    ########################
    

if __name__ == "__main__":
    main()
