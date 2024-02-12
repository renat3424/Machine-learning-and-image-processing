import numpy as np
import cv2
import matplotlib.pyplot as plt
from compvision3 import Canny

shapes = cv2.imread('coins_150.png')
shapes=cv2.resize(shapes,(300,300))
cv2.imshow('Original Image', shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
canny=Canny(shapes)
canny_edges=canny.Final()
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

def huf_circles(img):

    height, width = img.shape
    r_max=np.max((height, width))
    r_min=3
    num_thetas=120
    dtheta=3



    rhos = np.arange(r_min, r_max + 1, 1)
    thetas = np.deg2rad(np.arange(0, 360, dtheta))


    H = np.zeros((len(rhos), 3*len(rhos)+9, 3*len(rhos)+9), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)
    circle_candidates = []
    for r in range(len(rhos)):
        for t in range(num_thetas):
            circle_candidates.append((r, int(rhos[r] * np.cos(thetas[t])), int(rhos[r] * np.sin(thetas[t]))))
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for r, rcos_t, rsin_t in circle_candidates:
            x_center = len(rhos)+x - rcos_t
            y_center = len(rhos)+y - rsin_t
            H[r, x_center, y_center] += 1

    return H, rhos


def _circles(H, num_peaks):

    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)
        H1_idx = np.unravel_index(idx, H1.shape)
        indicies.append(H1_idx)


        idx_x, idx_y, idx_z = H1_idx
        H1[idx_y, idx_x, idx_z] = 0

    return indicies

def huf_circles_draw(img, indicies, rhos):

    for i in range(len(indicies)):

        rho = rhos[indicies[i][0]]
        xcenter = indicies[i][1]-len(rhos)
        ycenter = indicies[i][2]-len(rhos)


        cv2.circle(img, (xcenter, ycenter), rho, (0, 255, 0), 2)

H, rhos = huf_circles(canny_edges)
indicies = _circles(H,7)

huf_circles_draw(shapes, indicies, rhos)


cv2.imshow("Image with lines", shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()