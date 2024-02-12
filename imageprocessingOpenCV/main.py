import cv2
import numpy as np

from matplotlib import pyplot as plt


def gamma_channel(image, gamma, channel):
    img=image.copy()
    img[:,:,channel]=np.power(img[:,:,channel], gamma)
    return img

def new_image(image):
    img=cv2.imread(image).astype(np.float32)/255
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def gamma_correction(image):
    gammas=[0.25, 0.5, 1.5, 2]
    colors=["red", "green", "blue"]
    fig = plt.figure(figsize=(40, 40))
    rows = 3
    columns = len(gammas)
    for i in range(rows*columns):

        str1=colors[int(i%rows)]+" "+str(gammas[int(i/rows)])
        img=gamma_channel(image, gammas[int(i/rows)], int(i%rows))

        plt.imsave("images/people1/"+str1+".png", img)


handwriting=new_image("handwriting.jpg")
landscape=new_image("landscape.jpg")
people=new_image("people.jpg")
rentgenogram=new_image("rentgenogram.jpg")
gamma_correction(people)



