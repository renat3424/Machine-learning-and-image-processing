import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ORB:
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


    def count_ones(self, array):


        for i in range(len(array)):

            if array[np.array(range(i, i+12))%16].all()==True:

                return True



        return False


    def orientations(self, img, points, width):
        radius = int((width-1)/2)
        mask=self.create_circular_mask((radius, radius), radius, width)
        img = np.pad(img, (2*radius, 2*radius))
        mrows, mcols=mask.shape

        orientations = []
        for i in range(points.shape[0]):
            c0, r0 = points[i, :]
            m01, m10 = 0, 0
            for r in range(mrows):
                m01_temp = 0
                for c in range(mcols):
                    if mask[r, c]:

                        I = img[r0 + r, c0 + c]
                        m10 = m10 + I * (c - radius)
                        m01_temp = m01_temp + I
                m01 = m01 + m01_temp * (r - radius)
            angle=np.arctan2(m01, m10)
            if angle <= 0:
                angle = angle + np.pi * 2
            orientations.append(angle)

        return np.array(orientations)

    def orientation_calculation(self, in_mtrx, x_c, y_c, R):
        area, teta = [[0, 0]], 0
        w_borders = np.zeros(shape=(in_mtrx.shape[0] + 2 * R, in_mtrx.shape[1] + 2 * R))
        w_borders[R:-R, R:-R] = in_mtrx

        for r in range(1, R + 1):

            for _x in range(-r, r + 1):
                if r ** 2 - _x ** 2 < 0:
                    continue
                _y1 = int(np.round(np.sqrt(r ** 2 - _x ** 2)))
                _y2 = int(np.round(-np.sqrt(r ** 2 - _x ** 2)))
                if -r <= _y1 <= r:
                    area.append([_x, _y1])
                if -r <= _y2 <= r:
                    area.append([_x, _y2])
            for _y in range(-r, r + 1):
                if r ** 2 - _y ** 2 < 0:
                    continue
                _x1 = int(np.round(np.sqrt(r ** 2 - _y ** 2)))
                _x2 = int(np.round(-np.sqrt(r ** 2 - _y ** 2)))
                if -r <= _x1 <= r:
                    area.append([_x1, _y])
                if -r <= _x2 <= r:
                    area.append([_x2, _y])
        area = np.unique(area, axis=0)

        m00, m01, m10 = 0, 0, 0
        for p in area:
            I = w_borders[p[0] + x_c, p[1] + y_c]
            m00 += (p[0] ** 0) * (p[1] ** 0) * I
            m01 += (p[0] ** 0) * (p[1] ** 1) * I
            m10 += (p[0] ** 1) * (p[1] ** 0) * I
        teta_return = np.arctan2(m01, m10)
        if teta_return < 0:
            return np.arctan2(m01, m10) + np.pi * 2
        return np.arctan2(m01, m10)

    def orientations(self, img, points, R):
        orientations = []
        for k in points:
            orientations.append(self.orientation_calculation(img, k[0] + R, k[1] + R, R))
        return orientations



    def Fast(self, img, thresshold):
        keypoints=[]
        cross_idx = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
        circle_idx = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                               [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])
        for x in range(3, img.shape[0] - 3):
            for y in range(3, img.shape[1] - 3):
                Ip=img[x, y]
                if np.count_nonzero(Ip + thresshold < img[x + cross_idx[0, :], y + cross_idx[1, :]]) >= 3 or np.count_nonzero(Ip - thresshold > img[x + cross_idx[0, :], y + cross_idx[1, :]]) >= 3:
                    a=(Ip + thresshold < img[x + circle_idx[0, :], y + circle_idx[1, :]])
                    b=(Ip - thresshold > img[x + circle_idx[0, :], y + circle_idx[1, :]])
                    if self.count_ones(a) or self.count_ones(b):
                        keypoints.append([x, y])

        return keypoints
    def getGaussFilter(self):

        return  np.array([[1, 4,  7,  4,  1],[4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],[1, 4,  7,  4,  1]])/273
    def find_harris_points(self, input_img, k, window_size, points, N):

        offset = int(window_size / 2)
        x_range = input_img.shape[0] - offset
        y_range = input_img.shape[1] - offset

        dx = self.SobelFilter(input_img, "x")
        dy = self.SobelFilter(input_img, "y")
        Ixx = dx ** 2
        Ixy = dx * dy
        Iyy = dy ** 2
        Ixx= self.matrix_convolution(Ixx, self.getGaussFilter())
        Ixy = self.matrix_convolution(Ixy, self.getGaussFilter())
        Iyy = self.matrix_convolution(Iyy, self.getGaussFilter())
        rs=[]
        for point in points:

            x=point[0]
            y=point[1]
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1


            windowIxx = Ixx[start_x: end_x, start_y: end_y]
            windowIxy = Ixy[start_x: end_x, start_y: end_y]
            windowIyy = Iyy[start_x: end_x, start_y: end_y]


            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()


            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy


            r = det - k * (trace ** 2)

            rs.append(r)

        rs=np.array(rs)
        args=np.argsort(rs)[::-1]

        rs=rs[args]
        points=np.array(points)[args]

        return points[0:N]

    def create_circular_mask(self, center, radius, w):

        X, Y = np.ogrid[0:w, 0:w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist_from_center<=radius


    def Brief(self, img_input, keys_arr, orientations, patch_size, n):
        descriptors = []
        shift = int(patch_size / 2)
        gauss=self.matrix_convolution(img_input, self.getGaussFilter())
        mask=self.create_circular_mask((shift,shift), shift, patch_size)

        p_points=[]
        for i in range(keys_arr.shape[0]):
            c0, r0 = keys_arr[i, :]
            angle=orientations[i]
            S = np.zeros(shape=(2, n, 2))
            for j in range(n):
                if i == 0:
                    while True:
                        rand = np.random.normal(0, patch_size ** 2 / 25, 4).astype(int)%31

                        if mask[rand[0], rand[1]] and mask[rand[2], rand[3]]:
                            if not [rand[0], rand[1]] in p_points and not [rand[2], rand[3]] in p_points:
                                p_points.append([rand[0], rand[1]])
                                p_points.append([rand[2], rand[3]])
                                S[0, j] = [rand[0], rand[1]]
                                S[1, j] = [rand[2], rand[3]]
                                break
                else:
                    u1, u2 = p_points[2 * j], p_points[2 * j + 1]
                    S[0, j] = [u1[0], u1[1]]
                    S[1, j] = [u2[0], u2[1]]

            angles = np.zeros(30)
            for k in range(1, len(angles)):
                angles[k] = angles[k - 1] + 2 * np.pi / 30

            angle = (angle - (angle % 2 * np.pi / 30)) % 2 * np.pi


            _S1, _S2 = S[0].T, S[1].T
            S1 = (self.rotate_mtrx(angle) @ _S1).astype(int)
            S2 = (self.rotate_mtrx(angle) @ _S2).astype(int)


            bin_row = np.zeros(shape=n)
            for k in range(n):
                p1, p2 = S1.T[k], S2.T[k]
                print(p1, p2)
                if gauss[c0 + p1[0], r0 + p1[1]] < gauss[c0 + p2[0], r0 + p2[1]]:

                    bin_row[k] = 1
            descriptors.append(list(bin_row))


        return descriptors

    def brief_algorithm(self, in_mtrx, keys_arr, orientations, patch_size, n):
        descriptors, all_descriptors = [], np.zeros(shape=(len(keys_arr), 30, n))

        shift = int(patch_size / 2)
        frame = np.zeros(shape=(in_mtrx.shape[0] + patch_size - 1, in_mtrx.shape[1] + patch_size - 1))
        frame[shift:-shift, shift:-shift] = in_mtrx

        angles = np.zeros(30)
        for i in range(len(angles)):
            if i == 0:
                angles[i] = 0
            else:
                angles[i] = angles[i - 1] + 2 * np.pi / 30

        for k in range(len(keys_arr)):
            x_c, y_c = keys_arr[k][0] + shift, keys_arr[k][1] + shift
            teta_c = orientations[k]
            area = [[0, 0]]

            for r in range(1, shift + 1):
                for _x in range(x_c - r, x_c + r + 1):
                    if r ** 2 - (_x - x_c) ** 2 < 0:
                        continue
                    _y1 = int(np.round(np.sqrt(r ** 2 - (_x - x_c) ** 2) + y_c))
                    _y2 = int(np.round(-np.sqrt(r ** 2 - (_x - x_c) ** 2) + y_c))
                    if 0 <= _y1 < frame.shape[1]:
                        area.append([_x - x_c, _y1 - y_c])
                    if 0 <= _y2 < frame.shape[1]:
                        area.append([_x - x_c, _y2 - y_c])
                for _y in range(y_c - r, y_c + r + 1):
                    if r ** 2 - (_y - y_c) ** 2 < 0:
                        continue
                    _x1 = int(np.round(np.sqrt(r ** 2 - (_y - y_c) ** 2) + x_c))
                    _x2 = int(np.round(-np.sqrt(r ** 2 - (_y - y_c) ** 2) + x_c))
                    if 0 <= _x1 < frame.shape[0]:
                        area.append([_x1 - x_c, _y - y_c])
                    if 0 <= _x2 < frame.shape[0]:
                        area.append([_x2 - x_c, _y - y_c])
            area = np.unique(area, axis=0)

            pattern_points = []
            for i in range(n):
                while True:

                    rand = np.random.normal(0, patch_size ** 2 / 25, 4).astype(int)
                    if [rand[0], rand[1]] in area.tolist() and [rand[2], rand[3]] in area.tolist():
                        if not [rand[0], rand[1]] in pattern_points and not [rand[2], rand[3]] in pattern_points:
                            pattern_points.append([rand[0], rand[1]])
                            pattern_points.append([rand[2], rand[3]])
                            break

            S = np.zeros(shape=(2, n, 2))
            for i in range(n):
                u1, u2 = pattern_points[2 * i], pattern_points[2 * i + 1]
                S[0, i] = [u1[0], u1[1]]
                S[1, i] = [u2[0], u2[1]]

            min_dif, round_teta_c = 100000000, 0
            for a in angles:
                dif = np.abs(a - teta_c)
                if dif < min_dif:
                    min_dif = dif
                    round_teta_c = a
            teta_c = round_teta_c

            for a in range(len(angles)):
                _S1, _S2 = S[0].T, S[1].T
                S1 = (self.rotate_mtrx(angles[a]) @ _S1).astype(int)
                S2 = (self.rotate_mtrx(angles[a]) @ _S2).astype(int)
                bin_row = np.zeros(shape=n)
                for i in range(n):
                    p1, p2 = S1.T[i], S2.T[i]
                    if frame[x_c + p1[0], y_c + p1[1]] < frame[x_c + p2[0], y_c + p2[1]]:
                        bin_row[i] = 1
                if teta_c == angles[a]:
                    descriptors.append(list(bin_row))
                    all_descriptors[k, a] = bin_row
                else:
                    all_descriptors[k, a] = bin_row
        return descriptors, all_descriptors, pattern_points

    def rotate_mtrx(self, teta):
        return np.array([[np.cos(teta), np.sin(teta)], [-np.sin(teta), np.cos(teta)]])
    def write_in_file(self, file_name, array):
        f = open(file_name, 'w')

        for d in array:
            f.write(str(d) + '\n')


    def draw_dot(self, semitone, x, y,  size, color):
        shift = int(size / 2)
        for i in range(-shift, shift + 1):
            for j in range(-shift, shift + 1):
                if (0 > x + i or x + i >= semitone.shape[0]) or (0 > y + j or y + j >= semitone.shape[1]):
                    continue
                semitone[x + i, y + j] = color


    def draw_dots(self, image, harris_dots):

        for i in range(len(harris_dots)):
            self.draw_dot(image, harris_dots[i][0], harris_dots[i][1], 5, 255)







if __name__ == "__main__":

    orb=ORB()
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    im=np.array(Image.open("box_in_scene.png"))
    plt.imshow(im)

    halftone_image=im

    fig.add_subplot(1, 2, 2)

    plt.imshow(halftone_image, cmap="gray")
    plt.show()
    keypoints=orb.Fast(halftone_image, 5)
    print(len(keypoints))
    fig=plt.figure(1)
    mtr=np.zeros(halftone_image.shape)
    orb.draw_dots(mtr, keypoints)
    plt.imshow(mtr, cmap="gray")
    plt.show()
    harris_points=orb.find_harris_points(halftone_image, 0.04, 5, keypoints, 150)
    fig=plt.figure(2)
    halftone_image1=halftone_image.copy()
    orb.draw_dots(halftone_image1, harris_points)
    plt.imshow(halftone_image1, cmap="gray")
    plt.show()
    orients=orb.orientations(halftone_image, harris_points, 31)
    print(orients)
    descriptors=orb.brief_algorithm(halftone_image, harris_points, orients, 31, 256)[0]
    orb.write_in_file('descriptors1.txt', descriptors)










