import numpy as np

from matplotlib import pyplot as plt
import cv2

class Canny:
    def __init__(self, image):

        self.im = image
        self.imOr=self.im


        self.Gx = self.SobelFilter(self.im, "x")
        self.Gy = self.SobelFilter(self.im, "y")
        self.magnitude = np.hypot(self.Gx, self.Gy)
        degrees = np.degrees(np.arctan2(self.Gy, self.Gx))
        self.degrees = self.DegreesRound(degrees)
        self.nonmaximumsup = self.NonMaxSup(self.magnitude, self.degrees)
        self.Hyst = self.Hysteresis(self.nonmaximumsup)


    def Final(self):
        return self.Hyst
    def Magnitude(self):
        return self.magnitude
    def Plot(self):
        fig = plt.figure(1)
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.imOr, cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(self.im, cmap="gray")
        plt.show()
        fig = plt.figure(2)
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.Gx, cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(self.Gy, cmap="gray")
        plt.show()
        fig = plt.figure(3)
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.magnitude, cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(self.nonmaximumsup, cmap="gray")
        plt.show()
        plt.figure(4)
        plt.imshow(self.Hyst, cmap="gray")
        plt.show()

    def Gauss(self, x, y, sigma):
        return np.exp(-1 * (x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    def Filter(self, dim, sigma):
        filter = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                filter[i, j] = self.Gauss((dim - 1) / 2 - i, j - (dim - 1) / 2, sigma)

        return filter / filter.sum()

    def matrix_convolution(self, matrix, kernel):

        k_size = len(kernel)
        m_height, m_width = matrix.shape
        padded = np.pad(matrix, (k_size - 1, k_size - 1))


        output = []
        for i in range(m_height):
            for j in range(m_width):
                output.append(np.sum(padded[i:k_size + i, j:k_size + j] * kernel))

        output = np.array(output).reshape((m_height, m_width))
        return output

    def SobelFilter(self, img, direction):

        if direction=="x":
            gx=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            return  self.matrix_convolution(img, gx)
        if direction=="y":
            gy=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            return  self.matrix_convolution(img, gy)



    def GaussCore(self, dim, array, sigma):
        array1 = np.zeros((array.shape[0] + dim - 1, array.shape[1] + dim - 1, array.shape[2]))
        for i in range(int((dim - 1) / 2), int(array.shape[0] + (dim - 1) / 2)):
            for j in range(int((dim - 1) / 2), int(array.shape[1] + (dim - 1) / 2)):
                array1[i, j, 0] = array[i - int((dim - 1) / 2), j - int((dim - 1) / 2), 0]
                array1[i, j, 1] = array[i - int((dim - 1) / 2), j - int((dim - 1) / 2), 1]
                array1[i, j, 2] = array[i - int((dim - 1) / 2), j - int((dim - 1) / 2), 2]
        for i in range(int((dim - 1) / 2), int(array.shape[0] + (dim - 1) / 2)):
            for j in range(int((dim - 1) / 2), int(array.shape[1] + (dim - 1) / 2)):
                i1 = i - int((dim - 1) / 2)
                j1 = j - int((dim - 1) / 2)
                filter = self.Filter(dim, sigma)
                S1 = 0
                S2 = 0
                S3 = 0

                for i2 in range(dim):
                    for j2 in range(dim):
                        S1 += array1[i2 + i1, j2 + j1, 0] * filter[i2, j2]
                        S2 += array1[i2 + i1, j2 + j1, 1] * filter[i2, j2]
                        S3 += array1[i2 + i1, j2 + j1, 2] * filter[i2, j2]

                array1[i, j, 0] = int(S1)

                array1[i, j, 1] = int(S2)
                array1[i, j, 2] = int(S3)

        array2 = np.zeros(array.shape)
        for i in range(int((dim - 1) / 2), int(array.shape[0] + (dim - 1) / 2)):
            for j in range(int((dim - 1) / 2), int(array.shape[1] + (dim - 1) / 2)):
                array2[i - int((dim - 1) / 2), j - int((dim - 1) / 2), 0] = array1[i, j, 0]
                array2[i - int((dim - 1) / 2), j - int((dim - 1) / 2), 1] = array1[i, j, 1]
                array2[i - int((dim - 1) / 2), j - int((dim - 1) / 2), 2] = array1[i, j, 2]
        return array2
    def DegreesRound(self, degrees):
        degrees = degrees.astype(int)
        degrees[(degrees < 0)]+= 360
        degrees[(degrees > 337)&(degrees <= 22)] = 0
        degrees[(degrees > 22)&(degrees <= 67)] = 45
        degrees[(degrees > 67)&(degrees <= 112)] = 90
        degrees[(degrees > 112)&(degrees <= 157)] = 135
        degrees[(degrees > 157)&(degrees <= 202)] = 180
        degrees[(degrees > 202)&(degrees <= 247)] = 225
        degrees[(degrees > 247)&(degrees <= 292)] = 270
        degrees[(degrees > 292)&(degrees <= 337)] = 315
        return degrees


    def NonMaxSup(self, magnitude, degrees):
        NMS = np.zeros(magnitude.shape)

        for i in range(1, int(magnitude.shape[0]) - 1):
            for j in range(1, int(magnitude.shape[1]) - 1):
                if ((degrees[i, j] == 0) or (degrees[i, j] == 180)):

                    if (magnitude[i, j] >= magnitude[i+1, j] and magnitude[i, j] >= magnitude[i-1, j]):
                        NMS[i, j] = magnitude[i, j]
                    else:
                        NMS[i, j] = 0
                if ((degrees[i, j] == 90) or (degrees[i, j] == 270)):

                    if (magnitude[i, j] >= magnitude[i, j-1] and magnitude[i, j] >= magnitude[i, j+1]):
                        NMS[i, j] = magnitude[i, j]
                    else:
                        NMS[i, j] = 0
                if ((degrees[i, j] == 45) or (degrees[i, j] == 225)):

                    if (magnitude[i, j] >= magnitude[i + 1, j+1] and magnitude[i, j] >= magnitude[i - 1, j-1]):
                        NMS[i, j] = magnitude[i, j]
                    else:
                        NMS[i, j] = 0
                if ((degrees[i, j] == 135) or (degrees[i, j] == 315)):

                    if (magnitude[i, j] >= magnitude[i + 1, j-1] and magnitude[i, j] >= magnitude[i - 1, j+1]):
                        NMS[i, j] = magnitude[i, j]
                    else:
                        NMS[i, j] = 0

        return NMS
    def Hysteresis(self, img):
        print(img)
        highThreshold = 200
        lowThreshold = 100
        Hyst = np.copy(img)
        h = int(Hyst.shape[0])
        w = int(Hyst.shape[1])

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (Hyst[i, j] > highThreshold):
                    Hyst[i, j] = 255
                elif (Hyst[i, j] < lowThreshold):
                    Hyst[i, j] = 0
                else:
                    if ((Hyst[i - 1, j - 1] > highThreshold) or
                            (Hyst[i - 1, j] > highThreshold) or
                            (Hyst[i - 1, j + 1] > highThreshold) or
                            (Hyst[i, j - 1] > highThreshold) or
                            (Hyst[i, j + 1] > highThreshold) or
                            (Hyst[i + 1, j - 1] > highThreshold) or
                            (Hyst[i + 1, j] > highThreshold) or
                            (Hyst[i + 1, j + 1] > highThreshold)):
                        Hyst[i, j] = 255

        return Hyst


def img_tohalftone(image):
    halftone_image = (image[:, :, 0] / 3 + image[:, :, 1] / 3 + image[:, :, 2] / 3)

    halftone_image = halftone_image.astype(int)
    return halftone_image













if __name__ == '__main__':
    img = cv2.imread('8.jpg', 0)
    edges = cv2.Canny(img, 100, 200)
    fig = plt.figure()
    plt.imshow(edges, cmap="gray")

    plt.show()
