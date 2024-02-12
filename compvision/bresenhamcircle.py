import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

subplots = 0


def img_open(name):
    return Image.open(name)


def img_show(img_mtrx, title, isHalftone, isResultImage):
    if not isResultImage:
        global subplots
        subplots += 1
        if isHalftone:
            plt.subplot(1, 2, subplots)
            plt.title(title)
            plt.imshow(img_mtrx.astype('uint8'), cmap="gray")
        else:
            plt.subplot(1, 2, subplots)
            plt.title(title)
            plt.imshow(img_mtrx)
    else:
        if isHalftone:
            plt.figure()
            plt.title(title)
            plt.imshow(img_mtrx.astype('uint8'), cmap="gray")
        else:
            plt.figure()
            plt.title(title)
            plt.imshow(img_mtrx)


def img_to_np(image):
    return np.array(image)


def img_halftone(img_mtrx, ifRgb):
    if ifRgb:
        halftone_mtrx = np.zeros((len(img_mtrx), len(img_mtrx[0]), 3))
        for i in range(len(halftone_mtrx)):
            for j in range(len(halftone_mtrx[i])):
                for k in range(3):
                    halftone_mtrx[i, j, k] = np.mean(img_mtrx[i, j])
        return halftone_mtrx.astype(int)
    else:
        halftone_mtrx = np.zeros((len(img_mtrx), len(img_mtrx[0])))
        for i in range(len(halftone_mtrx)):
            for j in range(len(halftone_mtrx[i])):
                halftone_mtrx[i, j] = np.mean(img_mtrx[i, j])
        return halftone_mtrx.astype(int)


def gaus_filter(sigma, size):
    gaus_filter = np.zeros((size, size))
    shift = int(size / 2)
    for i in range(-shift, shift + 1):
        for j in range(-shift, shift + 1):
            gaus_filter[i + shift, j + shift] = 1 / (2 * np.pi * sigma ** 2) * np.exp(
                -(i ** 2 + j ** 2) / (2 * sigma ** 2))
    return gaus_filter / np.sum(gaus_filter), shift


def gauss_weight(x, y, sigma):
    return 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def img_blur(img_mtrx, filter_size, sigma, isHalftone):
    g_filter, shift = gaus_filter(sigma, filter_size)
    filtered_mtrx = np.copy(img_mtrx)
    # добавление строк
    if isHalftone:
        for i in range(shift):
            row = np.zeros((1, len(filtered_mtrx[0])))
            filtered_mtrx = np.concatenate((row, filtered_mtrx, row), axis=0)
            col = np.zeros((len(filtered_mtrx), 1))
            filtered_mtrx = np.concatenate((col, filtered_mtrx, col), axis=1)
    # размытие
    for i in range(shift, len(filtered_mtrx) - shift):
        for j in range(shift, len(filtered_mtrx[i]) - shift):
            neighbourhood = filtered_mtrx[i - shift: i + shift + 1, j - shift: j + shift + 1]
            sum = 0
            if isHalftone:
                for u in range(len(g_filter)):
                    for v in range(len(g_filter[u])):
                        sum += neighbourhood[u][v] * g_filter[u][v]
                filtered_mtrx[i][j] = sum
            else:
                for k in range(3):
                    for u in range(len(g_filter)):
                        for v in range(len(g_filter[u])):
                            sum += neighbourhood[u][v][k] * g_filter[u][v]
                        filtered_mtrx[i][j][k] = sum
    # удаление строк
    if isHalftone:
        for i in range(shift):
            filtered_mtrx = np.delete(filtered_mtrx, 0, axis=0)
            filtered_mtrx = np.delete(filtered_mtrx, len(filtered_mtrx) - 1, axis=0)
            filtered_mtrx = np.delete(filtered_mtrx, 0, axis=1)
            filtered_mtrx = np.delete(filtered_mtrx, len(filtered_mtrx[0]) - 1, axis=1)
    return filtered_mtrx.astype('uint8')


def get_brezenham_circle(in_mtrx, x, y):
    return [in_mtrx[x - 3, y], in_mtrx[x - 3, y + 1], in_mtrx[x - 2, y + 2], in_mtrx[x - 1, y + 3], in_mtrx[x, y + 3],
            in_mtrx[x + 1, y + 3], in_mtrx[x + 2, y + 2], in_mtrx[x + 3, y + 1], in_mtrx[x + 3, y],
            in_mtrx[x + 3, y - 1], in_mtrx[x + 2, y - 2], in_mtrx[x + 1, y - 3], in_mtrx[x, y - 3],
            in_mtrx[x - 1, y - 3], in_mtrx[x - 2, y - 2], in_mtrx[x - 3, y - 1]]


def fast_algorithm(halftone_mtrx, t, n):
    in_mtrx = np.copy(halftone_mtrx)
    keys_arr, keys_mtrx = [], np.zeros(shape=halftone_mtrx.shape)
    # проход по всем пикселям изображения
    for i in range(3, in_mtrx.shape[0] - 3):
        for j in range(3, in_mtrx.shape[1] - 3):
            brez_circle = get_brezenham_circle(in_mtrx, i, j)
            cur_I = in_mtrx[i, j]
            # проверка 1, 9 и 5, 13
            if ((brez_circle[0] > cur_I + t) and (brez_circle[8] < cur_I - t)) or (
                    (brez_circle[0] < cur_I - t) and (brez_circle[8] > cur_I + t)):
                continue
            if ((brez_circle[4] > cur_I + t) and (brez_circle[12] < cur_I - t)) or (
                    (brez_circle[4] < cur_I - t) and (brez_circle[12] > cur_I + t)):
                continue
            s = len(brez_circle)
            for u in range(s):  # каждая точка из окружности
                isMore, count = False, 0
                for v in range(n):  # смотрим для нее последовательность из n точек
                    if v == 0:
                        if brez_circle[(u + v) % s] > cur_I + t and isMore is False:
                            isMore, count = True, 1
                        if brez_circle[(u + v) % s] < cur_I - t and isMore is True:
                            isMore, count = False, 1
                        if brez_circle[(u + v) % s] > cur_I + t and isMore is True:
                            count = 1
                        if brez_circle[(u + v) % s] < cur_I - t and isMore is False:
                            count = 1
                    else:
                        if brez_circle[(u + v) % s] > cur_I + t and isMore is False:
                            break
                        if brez_circle[(u + v) % s] > cur_I + t and isMore is True:
                            count += 1
                        if brez_circle[(u + v) % s] < cur_I - t and isMore is True:
                            break
                        if brez_circle[(u + v) % s] < cur_I - t and isMore is False:
                            count += 1
                if count == n:
                    keys_arr.append([i, j])
                    keys_mtrx[i, j] = 255
                    break
    return keys_arr, keys_mtrx


def harris_criterion(in_mtrx, keys_arr, sigma, k, N):
    R_arr = {}
    sobel_gy, sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # проход по всем точкам из FAST
    for p in keys_arr:
        M = np.zeros(shape=(2, 2))
        # проход по точкам из окна 5х5
        gaussian = gaus_filter(sigma, 5)[0]
        for i in range(-2, 3):
            for j in range(-2, 3):
                _x, _y = p[0] + i, p[1] + j
                # подсчет частных производных в этой точке
                Ix, Iy = 0, 0
                neighborhood = in_mtrx[_x - 1: _x + 2, _y - 1: _y + 2]
                for u in range(len(sobel_gx)):
                    for v in range(len(sobel_gx[u])):
                        Ix += sobel_gx[u][v] * neighborhood[u][v]
                        Iy += sobel_gy[u][v] * neighborhood[u][v]
                # вычисление M
                A = np.array([[Ix ** 2, Ix * Iy], [Ix * Iy, Iy ** 2]])
                M = M + gaussian[i + 2, j + 2] * A
        # подсчет R
        l1, l2 = np.linalg.eigvals(M)[0], np.linalg.eigvals(M)[1]
        det_M, trace_M = l1 * l2, l1 + l2
        R = det_M - k * (trace_M ** 2)
        if R > 0:
            R_arr[R] = [p[0], p[1]]
    out = dict(sorted(R_arr.items(), reverse=True, key=lambda x: x[0]))
    return list(out.values())[:N]


def draw_point(in_mtrx, x, y, size, color):
    shift = int(size / 2)
    for i in range(-shift, shift + 1):
        for j in range(-shift, shift + 1):
            if (0 > x + i or x + i >= in_mtrx.shape[0]) or (0 > y + j or y + j >= in_mtrx.shape[1]):
                continue
            in_mtrx[x + i, y + j] = color


def orientation_calculation(in_mtrx, x_c, y_c, R):
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
    # вычисляем моменты
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


def get_orientations(keys_arr, halftone_mtrx, R):
    orientations = []
    for k in keys_arr:
        orientations.append(orientation_calculation(halftone_mtrx, k[0] + R, k[1] + R, R))
    return orientations


def rotate_mtrx(teta):
    return np.array([[np.cos(teta), np.sin(teta)], [-np.sin(teta), np.cos(teta)]])


def brief_algorithm(in_mtrx, keys_arr, orientations, patch_size, n, pattern_points=None):
    descriptors, all_descriptors = [], np.zeros(shape=(len(keys_arr), 30, n))
    # добавляем нулевую рамку размером shift к матрице изображению
    shift = int(patch_size / 2)
    frame = np.zeros(shape=(in_mtrx.shape[0] + patch_size - 1, in_mtrx.shape[1] + patch_size - 1))
    frame[shift:-shift, shift:-shift] = in_mtrx
    # создаем набор углов с шагом в 15 градусов
    angles = np.zeros(30)
    for i in range(len(angles)):
        if i == 0:
            angles[i] = 0
        else:
            angles[i] = angles[i - 1] + 2 * np.pi / 30
    # проходимся по всем особым точкам
    for k in range(len(keys_arr)):
        x_c, y_c = keys_arr[k][0] + shift, keys_arr[k][1] + shift
        teta_c = orientations[k]
        area = [[0, 0]]
        # рассматриваем область радиуса shift
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
        # генерация паттерна (256 пар точек), если его нет
        if pattern_points is None:
            pattern_points = []
            for i in range(n):
                while True:
                    # принадлежат area, не принадлежат pattern_points чтобы были пары точек не повторялись,
                    rand = np.random.normal(0, patch_size ** 2 / 25, 4).astype(int)
                    if [rand[0], rand[1]] in area.tolist() and [rand[2], rand[3]] in area.tolist():
                        if not [rand[0], rand[1]] in pattern_points and not [rand[2], rand[3]] in pattern_points:
                            pattern_points.append([rand[0], rand[1]])
                            pattern_points.append([rand[2], rand[3]])
                            break
        # заполняем S матрицу 256 парами точек
        S = np.zeros(shape=(2, n, 2))
        for i in range(n):
            u1, u2 = pattern_points[2 * i], pattern_points[2 * i + 1]
            S[0, i] = [u1[0], u1[1]]
            S[1, i] = [u2[0], u2[1]]
        # округляем ориентацию особой точки до нужного угла
        min_dif, round_teta_c = 100000000, 0
        for a in angles:
            dif = np.abs(a - teta_c)
            if dif < min_dif:
                min_dif = dif
                round_teta_c = a
        teta_c = round_teta_c
        # вычисление дескрипторов для всех углов
        for a in range(len(angles)):
            _S1, _S2 = S[0].T, S[1].T
            S1 = (rotate_mtrx(angles[a]) @ _S1).astype(int)
            S2 = (rotate_mtrx(angles[a]) @ _S2).astype(int)
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


def get_all_descriptors(img_name, t_FAST, N_Harris, sigma_Harris, point_size, fromAFile, pattern_points=None):
    if fromAFile:
        if pattern_points is None:
            return np.load('keys_arr1.npy'), np.load('orientations1.npy'), np.load('desc1.npy'), np.load(
                'all_desc1.npy'), np.load('r_pattern_points1.npy')
        else:
            return np.load('keys_arr2.npy'), np.load('orientations2.npy'), np.load('desc2.npy'), np.load(
                'all_desc2.npy')
    # загрузка изображения
    img = img_open(img_name)
    img_mtrx = img_to_np(img)
    # перевод в полутон
    halftone_mtrx = img_halftone(img_mtrx, False)
    # FAST алгоритм (Детектирование особых (ключевых) точек)
    keys_arr, keys_mtrx = fast_algorithm(halftone_mtrx, t_FAST, 12)
    print(f'Всего особых точек {img_name} после алгоритма FAST: {len(keys_arr)}')
    # критерий угловых точек Харриса
    keys_arr1 = harris_criterion(halftone_mtrx, keys_arr, sigma_Harris, 0.04, int(N_Harris * len(keys_arr)))
    print(f'Всего особых точек {img_name} после критерия Харриса и фильтрации: {len(keys_arr1)}')
    # вычисление ориентаций у особых точек
    orientations = get_orientations(keys_arr1, halftone_mtrx, 31)
    # отрисовка ключевых точек
    keys_mtrx1 = img_halftone(img_mtrx, True)
    for i in range(len(keys_arr1)):
        draw_point(keys_mtrx1, keys_arr1[i][0], keys_arr1[i][1], point_size, [0, 255, 0])
    img_show(keys_mtrx1, f'Особые точки {img_name} после Harris ', False, False)
    # BRIEF
    blured_mtrx = img_blur(halftone_mtrx, 5, 0.5, True)
    desc, all_desc, r_pattern_points = brief_algorithm(blured_mtrx, keys_arr1, orientations, 31, 256, pattern_points)
    if pattern_points is None:
        np.save('keys_arr1', keys_arr1)
        np.save('orientations1', orientations)
        np.save('desc1', desc)
        np.save('all_desc1', all_desc)
        np.save('r_pattern_points1', r_pattern_points)
        return keys_arr1, orientations, desc, all_desc, r_pattern_points
    else:
        np.save('keys_arr2', keys_arr1)
        np.save('orientations2', orientations)
        np.save('desc2', desc)
        np.save('all_desc2', all_desc)
        np.save('r_pattern_points2', r_pattern_points)
        return keys_arr1, orientations, desc, all_desc


def img_connection(img_name1, img_name2):
    # загрузка изображения
    img = img_open(img_name1)
    img_mtrx1 = img_halftone(img_to_np(img), True)
    img = img_open(img_name2)
    img_mtrx2 = img_halftone(img_to_np(img), True)

    h1, w1, h2, w2 = img_mtrx1.shape[0], img_mtrx1.shape[1], img_mtrx2.shape[0], img_mtrx2.shape[1]
    if h1 >= h2:
        out = np.zeros(shape=(h1, w1 + w2, 3))
        out[0:h1, 0:w1, 0:3] = img_mtrx1
        out[0:h2, w1:w1 + w2, 0:3] = img_mtrx2
    else:
        out = np.zeros(shape=(h2, w1 + w2, 3))
        out[0:h1, 0:w1, 0:3] = img_mtrx1
        out[0:h2, w1:w1 + w2, 0:3] = img_mtrx2
    return out


def Hamming_distance(desc1, desc2):
    return np.sum(np.absolute(desc1 - desc2))


def descriptors_enumeration(keys_arr1, keys_arr2, all_desc1, desc2, threshold):
    similar_points = []

    for i in range(len(all_desc1)):
        print(f'{i + 1} / {len(all_desc1)}')
        possible_points = {}

        for j in range(len(all_desc1[i])):

            for k in range(len(desc2)):
                d1, d2 = all_desc1[i, j], desc2[k]
                r = Hamming_distance(d1, d2)
                if r <= threshold:
                    possible_points[r] = [keys_arr1[i], keys_arr2[k]]

        if len(possible_points) == 0:
            continue
        possible_points = sorted(possible_points.items(), key=lambda x: x[0])[0:2]
        if len(possible_points) == 1:
            p1, p2 = possible_points[0][1][0], possible_points[0][1][1]
            similar_points.append([p1, p2])
            print(f'{p1}, {p2}, r = {possible_points[0][0]}')
        else:
            r1, r2 = possible_points[0][0], possible_points[1][0]
            p11, p12 = possible_points[0][1][0], possible_points[0][1][1]
            if r1 / r2 < 0.8:
                similar_points.append([p11, p12])
                print(f'{p11}, {p12}, r1 / r2 = {r1 / r2}')
    return similar_points


def parameter_estimation_RANSAC(N, similar_points):
    inlier = 0
    best_x = 0
    for i in range(N):
        rand_indexes = np.random.randint(0, len(similar_points), 3)
        pair1, pair2, pair3 = similar_points[rand_indexes[0]], similar_points[rand_indexes[1]], similar_points[
            rand_indexes[2]]
        A = np.array([
            [pair1[0, 0], pair1[0, 1], 0, 0, 1, 0],
            [0, 0, pair1[0, 0], pair1[0, 1], 0, 1],
            [pair2[0, 0], pair2[0, 1], 0, 0, 1, 0],
            [0, 0, pair2[0, 0], pair2[0, 1], 0, 1],
            [pair3[0, 0], pair3[0, 1], 0, 0, 1, 0],
            [0, 0, pair3[0, 0], pair3[0, 1], 0, 1]
        ])
        b = np.array([
            pair1[1, 0],
            pair1[1, 1],
            pair2[1, 0],
            pair2[1, 1],
            pair3[1, 0],
            pair3[1, 1]
        ])
        if np.linalg.det(A.T @ A) == 0:
            continue
        x = np.linalg.inv(A.T @ A) @ A.T @ b
        inlier_i = 0
        for pair in similar_points:
            p_query, p_test = pair[0], pair[1]
            M, T = np.array([[x[0], x[1]], [x[2], x[3]]]), np.array([x[4], x[5]])
            p_check = (M @ p_query + T).astype(int)
            if np.array_equiv(p_check, p_test):
                inlier_i += 1
        if inlier_i > inlier:
            inlier = inlier_i
            best_x = x
    return best_x


distance_threshold = 35
p1_percentage, p2_percentage = 0.8, 0.4

keys_arr1, orientations1, desc1, all_desc1, pattern_points = get_all_descriptors('img/box0.5.png', 4, p1_percentage, 1,
                                                                                 1, True)

keys_arr2, orientations2, desc2, all_desc2 = get_all_descriptors('img/box_in_scene.png', 4, p2_percentage, 1, 3, True,
                                                                 pattern_points)

similar_points = descriptors_enumeration(keys_arr1, keys_arr2, all_desc1, desc2, distance_threshold)
np.save('similar_points_' + str(distance_threshold), similar_points)

similar_points = np.load('similar_points_' + str(distance_threshold) + '.npy')
con_mtrx = img_connection('box0.5.png', 'box_in_scene.png')
for pair in similar_points:
    p1, p2 = pair[0], pair[1]
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1] + 162
    color = np.random.choice(range(256), size=3)
    draw_point(con_mtrx, x1, y1, 3, color)
    draw_point(con_mtrx, x2, y2, 3, color)
    if x1 > x2:
        for x in range(x2, x1 + 1):
            y = int(np.round(((x - x1) / (x2 - x1)) * (y2 - y1) + y1))
            draw_point(con_mtrx, x, y, 1, color)
    else:
        for x in range(x1, x2 + 1):
            y = int(np.round(((x - x1) / (x2 - x1)) * (y2 - y1) + y1))
            draw_point(con_mtrx, x, y, 1, color)

best_x = list(parameter_estimation_RANSAC(30, similar_points))
m11, m12, m21, m22 = best_x[0], best_x[1], best_x[2], best_x[3]
t1, t2 = best_x[4], best_x[5]
M = np.array([[m11, m12], [m21, m22]])
T = np.array([t1, t2])

img_query = img_open('box0.5.png')
img_mtrx = img_to_np(img_query)
halftone_query = img_halftone(img_mtrx, False)
polygon_query_points = []
h, w = len(halftone_query), len(halftone_query[0])
for i in range(h):
    for j in range(w):
        if i == 0 or i == h - 1 or j == 0 or j == w - 1:
            polygon_query_points.append([i, j])

for p_query in polygon_query_points:
    p_test = (M @ p_query + T).astype(int)
    draw_point(con_mtrx, p_test[0], p_test[1] + 162, 1, [0, 255, 0])

fig = plt.figure()
plt.imshow(con_mtrx.astype('uint8'), cmap="gray")

plt.show()














