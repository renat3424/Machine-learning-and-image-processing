import cv2
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt

def new_image(image):
    img=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    return img

def filter(image, coef):
    M, N = image.shape

    H = np.zeros((M, N), dtype=np.float32)

    D0 = coef

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if D <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0
    return H


def filterG(image, coef):


    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    D0 = coef
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = np.exp(-D ** 2 / (2 * D0 * D0))
    return H
def filters(image, coef, type="low", f="G"):

    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    if f=="G":
        H=filterG(image, coef)
    else:
        H =filter(image, coef)
    if type=="low":
        H=H
    else:
        H=1-H
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))
    plt.imsave("images/handwriting5/" +type +str(coef)+str(f) + "perfect.png", g, cmap="gray")







handwriting=new_image("handwriting.jpg")
landscape=new_image("landscape.jpg")
people=new_image("people.jpg")
rentgenogram=new_image("rentgenogram.jpg")


plt.imsave("images/handwriting5/" + "gray.png", people, cmap="gray")

array=[10, 15, 50, 150]
for i in array:
    filters(handwriting, i, "high", "I")
    filters(handwriting, i, "low", "I")
    filters(handwriting, i, "high", "G")
    filters(handwriting, i, "low", "G")
