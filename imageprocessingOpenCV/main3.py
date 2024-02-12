import cv2
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt

def new_image(image):
    img=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    return img


def morphologic_elements(image):
    plt.imsave("images/people4/" + "gray.png", image, cmap="gray")
    sizes = [5, 7]
    for size in sizes:
        kernal ={
            'rect':cv2.getStructuringElement(cv2.MORPH_RECT, (size,size)),
            'cross':cv2.getStructuringElement(cv2.MORPH_CROSS, (size,size)),
            'ell':cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
        }

        for ker in kernal:
            erosian = cv2.erode(image, kernal[ker])

            plt.imsave("images/people4/" + f'e{ker}{size}.png', erosian, cmap="gray")

        for ker in kernal:
            dilate = cv2.erode(image, kernal[ker])
            plt.imsave("images/people4/" + f'd{ker}{size}.png', dilate, cmap="gray")

        for ker in kernal:
            MORPH_OPEN = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernal[ker])
            plt.imsave("images/people4/" + f'o{ker}{size}.png', MORPH_OPEN, cmap="gray")
        for ker in kernal:
            MORPH_CLOSE = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal[ker])
            plt.imsave("images/people4/" + f'c{ker}{size}.png', MORPH_CLOSE, cmap="gray")

        for ker in kernal:
            MORPH_TOPHAT = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernal[ker])
            plt.imsave("images/people4/" + f't{ker}{size}.png', MORPH_TOPHAT, cmap="gray")
        for ker in kernal:
            MORPH_BLACKHAT = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernal[ker])
            plt.imsave("images/people4/" + f'b{ker}{size}.png', MORPH_BLACKHAT, cmap="gray")




handwriting=new_image("handwriting.jpg")
landscape=new_image("landscape.jpg")
people=new_image("people.jpg")
rentgenogram=new_image("rentgenogram.jpg")
morphologic_elements(people)