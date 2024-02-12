import cv2
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt

def new_image(image):
    img=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    return img


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def sobel_edge_detector(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    return grad_norm
def binary_images(image):
    plt.imsave("images/handwriting3/" + "gray.png", image, cmap="gray")
    _, imgThres=cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.imsave("images/handwriting3/" + "grayOtsu.png", imgThres, cmap="gray")
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    sobelxy = sobel_edge_detector(img_blur)
    print(mse(image, sobelxy))
    plt.imsave("images/handwriting3/" + "sobel.png", sobelxy, cmap="gray")
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    print(mse(image, edges))
    plt.imsave("images/handwriting3/" + "edges.png", edges, cmap="gray")
    for i in range(-2, 3):

        t=cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, i)
        plt.imsave("images/handwriting3/" + "grayGauss"+str(i)+".png", t, cmap="gray")
        print(mse(image, t))





handwriting=new_image("handwriting.jpg")
landscape=new_image("landscape.jpg")
people=new_image("people.jpg")
rentgenogram=new_image("rentgenogram.jpg")
binary_images(handwriting)