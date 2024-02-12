import numpy as np
import cv2
import matplotlib.pyplot as plt
from compvision3 import Canny


shapes = cv2.imread('train+tracks.jpg')
cv2.imshow('Original Image', shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
canny=Canny(shapes)
canny_edges=canny.Final()
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


def huf_lines(img):

    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height ** 2 + width ** 2))
    rhos = np.arange(0, img_diagonal + 1, 1)
    thetas = np.deg2rad(np.arange(-90, 180, 1))


    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])))
            H[rho, j] += 1

    return H, rhos, thetas





def huf_max(H, num_peaks):

    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)
        H1_idx = np.unravel_index(idx, H1.shape)
        indicies.append(H1_idx)


        idx_y, idx_x = H1_idx
        H1[idx_y, idx_x] = 0

    return indicies



def plot_huf(H):

    fig = plt.figure(figsize=(10, 10))


    plt.imshow(H, cmap='gray')

    plt.show()




def huf_lines_draw(img, indicies, rhos, thetas):

    for i in range(len(indicies)):

        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

H, rhos, thetas = huf_lines(canny_edges)
indicies = huf_max(H,10)
plot_huf(H)
huf_lines_draw(shapes, indicies, rhos, thetas)


cv2.imshow("Image with lines", shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
