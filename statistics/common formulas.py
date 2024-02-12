import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def mean(array):
    return array.sum()/array.size

def biased_var(array):
    return np.square(array).sum()/array.size-np.square(mean(array))

def unbiased_var(array):
    return array.size/(array.size-1)*biased_var(array)

def standart_deviation(array):
    return np.sqrt(unbiased_var(array))

def asymmetry_coef(array):
    return np.power((array-mean(array)), 3).sum()/array.size/np.power(biased_var(array), 3/2)

def excess_coef(array):
    return np.power((array-mean(array)), 4).sum()/array.size/np.power(biased_var(array), 2)-3

def median(array):
    num=int((array.size-1)/2)
    if((array.size-1)%2==0):
        return array[num]
    else:
        return (array[num]+array[num+1])/2




sample=np.sort(np.loadtxt("r3z1.csv", delimiter=",", dtype=np.str0)[1:].astype("float32"))

plt.style.use(["science", "notebook", "grid"])

sample_size=sample.size
sample_max=sample[-1]
sample_min=sample[0]
sample_range=sample_max-sample_min
sample_mean=mean(sample)
sample_bvar=biased_var(sample)
sample_ubvar=unbiased_var(sample)
sample_std=standart_deviation(sample)
sample_asymmetry=asymmetry_coef(sample)
sample_excess=excess_coef(sample)
sample_med=median(sample)


print(sample, "\n")
print("Объем: %.2f" %sample_size,"\n"
      "Максимум: %.2f" %sample_max,"\n",
      "Минимум: %.2f" %sample_min,"\n",
      "Размах: %.2f" %sample_range,"\n",
      "Среднее: %.2f" %sample_mean,"\n",
      "Смещенная Дисперсия: %.2f" %sample_bvar,"\n",
      "Несмещенная Дисперсия: %.2f" %sample_ubvar,"\n",
      "Стандартное Отклонение: %.2f" %sample_std,"\n",
      "Ассиметрия: %.2f" %sample_asymmetry,"\n",
      "Эксцесс: %.2f" %sample_excess,"\n",
      "Медиана: %.2f" %sample_med

      )

k=int(sample_size/10)-1

plt.figure(0, figsize=(10,4))

hist=plt.hist(sample, bins=k, density=True, edgecolor="black")
plt.xlabel('Интервалы')
plt.ylabel('Плотности частот')
modearg1=hist[0].argmax()
modearg2=modearg1+1
textstr=str.format("По гистограмме было получено, что %.2f<=мода<=%.2f" %(hist[1][modearg1], hist[1][modearg2]))

plt.ylim(top=0.2)

print(textstr)


y = ((1 / (np.sqrt(2 * np.pi) * sample_std)) *
     np.exp(-0.5 * (1 / sample_std * (sample - sample_mean))**2))
plt.plot(sample, y, '--')


F=np.arange(sample_size)/sample_size
plt.figure(1, figsize=(10,3.5))
plt.plot(sample, F, '-',  drawstyle='steps-pre')
plt.show()




