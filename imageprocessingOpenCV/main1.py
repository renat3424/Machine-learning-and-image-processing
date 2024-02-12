import cv2
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt


def new_image(image):
    img=cv2.imread(image).astype(np.float32)/255
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def noise_images(image):
    gausimage=random_noise(image, mode="gaussian", mean=0.05, var=0.02)

    plt.imsave("images/rentgen2/" + "gauss.png", gausimage)

    plt.imsave("images/rentgen2/" + "GgaussBlur3.png", cv2.GaussianBlur( gausimage, (3,3), 3))
    plt.imsave("images/rentgen2/" + "GgaussBlur5.png", cv2.GaussianBlur( gausimage, (5,5), 3))
    plt.imsave("images/rentgen2/" + "GmedianBlur3.png", cv2.medianBlur( np.uint8(gausimage*255), 3))
    plt.imsave("images/rentgen2/" + "GmedianBlur5.png", cv2.medianBlur( np.uint8(gausimage*255), 5))
    poisson = random_noise(image, mode="poisson")


    plt.imsave("images/rentgen2/" + "poisson.png", poisson)

    plt.imsave("images/rentgen2/" + "PgaussBlur3.png", cv2.GaussianBlur( poisson, (3,3), 3))
    plt.imsave("images/rentgen2/" + "PgaussBlur5.png", cv2.GaussianBlur( poisson, (5,5), 3))
    plt.imsave("images/rentgen2/" + "PmedianBlur3.png", cv2.medianBlur( np.uint8(poisson*255), 3))
    plt.imsave("images/rentgen2/" + "PmedianBlur5.png", cv2.medianBlur( np.uint8(poisson*255), 5))
    salt_pepper = random_noise(image, mode="s&p", amount=0.03)

    plt.imsave("images/rentgen2/" + "SgaussBlur3.png", cv2.GaussianBlur(salt_pepper, (3, 3), 3))
    plt.imsave("images/rentgen2/" + "SgaussBlur5.png", cv2.GaussianBlur(salt_pepper, (5, 5), 3))
    plt.imsave("images/rentgen2/" + "SmedianBlur3.png", cv2.medianBlur(np.uint8(salt_pepper*255), 3))
    plt.imsave("images/rentgen2/" + "SmedianBlur5.png", cv2.medianBlur(np.uint8(salt_pepper*255), 5))
    plt.imsave("images/rentgen2/" + "salt_pepper.png", salt_pepper)




handwriting=new_image("handwriting.jpg")
landscape=new_image("landscape.jpg")
people=new_image("people.jpg")
rentgenogram=new_image("rentgenogram.jpg")



noise_images(rentgenogram)