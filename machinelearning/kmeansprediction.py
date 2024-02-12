import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

digits = load_digits()
subplots = 0


def img_open(name):
    return Image.open(name)


def img_to_np(image):
    return np.array(image)


def img_show(img_mtrx, title, isHalftone, isResultImage):
    if not isResultImage:
        global subplots
        subplots += 1
        if isHalftone:
            plt.subplot(1, 1, subplots)
            plt.title(title)
            plt.imshow(img_mtrx.astype('uint8'), cmap="gray")
        else:
            plt.subplot(1, 1, subplots)
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


def euclid_distance(x, y):
    sum = 0
    for i in range(x.shape[0]):
        sum += (x[i] - y[i]) ** 2
    return np.sqrt(sum)


def E(clusters_x, centroids):
    sum = 0
    for j in range(len(centroids)):
        c_j, mu_j = clusters_x[j], centroids[j]
        for x in c_j:
            sum += euclid_distance(x, mu_j)
    return sum


def k_means(data, target, target_names, N, centroid_generation, data_type):
    centroids, prev_centroids, clusters_x, clusters_t = [], [], [], []
    # инициализация центроидов
    if centroid_generation == 'Рандом_из_выборки':
        for i in range(target_names.shape[0]):
            rand_ind = np.random.randint(0, data.shape[0])
            centroids.append(data[rand_ind])
    if centroid_generation == 'Рандом':
        for i in range(target_names.shape[0]):
            if data_type == 'Обычный':
                centroids.append(np.random.randint(0, 17, data.shape[1]))  # для обычного вектора характеристик
            if data_type == 'Гистограммы_интенсивностей':
                centroids.append(np.random.randint(0, 64, data.shape[1]))  # для гистограмм интенсивностей
            if data_type == 'Гистограммы_LBP':
                centroids.append(np.random.randint(0, 64, data.shape[1]))  # для LBP гистограмм
    if centroid_generation == 'Самые_дальние':
        # подсчет центроида для всех векторов характеристик
        centroid = np.zeros(shape=data.shape[1])
        for x in data:
            centroid = centroid + x / data.shape[0]
        # нахождение 10 дальних
        _centroids = {}
        for x in data:
            _centroids[euclid_distance(x, centroid)] = x
        _centroids = dict(sorted(_centroids.items(), key=lambda x: x[0], reverse=True)[:10])
        centroids = list(_centroids.values())
    # заполнение кластеров и пересчет центроидов
    for n in range(N):
        clusters_x = [[], [], [], [], [], [], [], [], [], []]
        clusters_t = [[], [], [], [], [], [], [], [], [], []]
        # для каждого вектора характеристик считаем минимальное расстояние до кластера
        for i in range(data.shape[0]):
            x, t = data[i], target[i]
            distances = []
            for mu in centroids:
                distances.append(euclid_distance(x, mu))
            c = np.argmin(distances)
            # добавление вектора и метки в кластер
            clusters_x[c].append(x)
            clusters_t[c].append(t)
        print(f'{n + 1}/{N}\nE = {E(clusters_x, centroids)}')
        # пересчет центроидов
        for i in range(len(clusters_x)):
            new_centroid = np.zeros(shape=data.shape[1])
            c_i = clusters_x[i]
            for x_i in c_i:
                new_centroid = new_centroid + x_i / len(c_i)
            centroids[i] = new_centroid
        # проверка на неизменность центроидов
        if n != 0:
            if np.array_equiv(centroids, prev_centroids):
                print('Новые центроиды совпали с предыдущими')
                return clusters_x, clusters_t, centroids
        prev_centroids = list(np.copy(centroids))
    return clusters_x, clusters_t, centroids


def get_histograms(data):
    histograms = []
    for x in data:
        histogram = np.zeros(17)
        for intensity in x:
            histogram[int(intensity)] += 1
        histograms.append(histogram)
    return np.array(histograms)


def from2_to10(bin_code):
    pow, sum = len(bin_code) - 1, 0
    for b in bin_code:
        sum += b * (2 ** pow)
        pow -= 1
    return sum


def get_lbp_histograms(images):
    histograms = []
    for img_mtrx in images:
        histogram = np.zeros(shape=256)
        h, w = img_mtrx.shape[0], img_mtrx.shape[1]
        # обрамление картинки
        _img_mtrx = np.zeros(shape=(h + 2, w + 2))
        _img_mtrx[1:_img_mtrx.shape[0] - 1, 1:_img_mtrx.shape[1] - 1] = img_mtrx
        # проход по всем пикселям картинки
        for i in range(1, _img_mtrx.shape[0] - 1):
            for j in range(1, _img_mtrx.shape[1] - 1):
                # бинарного кода для каждого пикселя
                bin_code = np.zeros(shape=8)
                area = np.array(
                    [_img_mtrx[i - 1, j - 1], _img_mtrx[i - 1, j], _img_mtrx[i - 1, j + 1], _img_mtrx[i, j + 1],
                     _img_mtrx[i + 1, j + 1], _img_mtrx[i + 1, j], _img_mtrx[i + 1, j - 1], _img_mtrx[i, j - 1]])
                for k in range(area.shape[0]):
                    if area[k] > _img_mtrx[i, j]:
                        bin_code[k] = 1
                # добавление числа в гистограмму изображения
                num = from2_to10(np.flip(bin_code))
                histogram[int(num)] += 1
        histograms.append(histogram)
    return histograms


# начальные данные
DESCR = digits.DESCR  # описание набора данных
images = digits.images  # массив из 1797 изображений, размер массива 1797х8х8
target = digits.target  # массив из меток изображений, 1797 элементов, значения от 0 до 9
target_names = digits.target_names  # массив имен меток, 10 элементов от 0 до 9
data = digits.data  # массив из “вытянутых” в строку 1797 изображений, размер 1797х64

# выбор типа вектора характеристик
data_type = 'Гистограммы_LBP'  # Обычный, Гистограммы_интенсивностей, Гистограммы_LBP
centroid_generation = 'Рандом_из_выборки'  # Рандом, Рандом_из_выборки, Самые_дальние
print(f'Вектор характеристик: {data_type}\nГенерация центроида: {centroid_generation}')

# получение гистограмм
histograms = []
if data_type == 'Гистограммы_интенсивностей':
    histograms = get_histograms(data)  # размер массива 1797х17
if data_type == 'Гистограммы_LBP':
    histograms = get_lbp_histograms(images)  # размер массива 1797х256
if data_type != 'Обычный':
    data = np.copy(histograms)

# получение кластеров и центроидов
clusters_x, clusters_t, centroids = k_means(data, target, target_names, 100, centroid_generation,
                                            data_type)  # Рандом, Рандом_из_выборки, Самые_дальние
for i in range(len(clusters_t)):
    print(f'{i + 1}/{len(clusters_t)}:\n{clusters_t[i]}')

# проверка модели
correct = 0
incorrect_predicted = np.zeros(shape=10)
for i in range(data.shape[0]):
    x, t = data[i], target[i]
    distances = []
    for mu in centroids:
        distances.append(euclid_distance(x, mu))
    c = np.argmin(distances)
    t_predicted = np.argmax(np.bincount(clusters_t[c]))
    if t_predicted == t:
        correct += 1
    else:
        incorrect_predicted[t] += 1
print(f'Процент угаданных: {np.round((correct / data.shape[0]) * 100, 2)}%')

# вывод информации о неугаданных цифрах
print(f'Количество раз, когда каждая цифра была не угадана:')
for i in range(incorrect_predicted.shape[0]):
    print(f'{i}: {incorrect_predicted[i]}')

plt.show()
