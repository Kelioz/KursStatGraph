import numpy as np
from math import *
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd


P = 0.92  # Доверительная вероятность
q = 0.05  # Уровень значимости
C = 0.01  # Цена деления прибора С, мм

path = 'qwe.xlsx'
# Изменение с запятой на точку для преобразования str в float
data = pd.read_excel(path)


# создание массива из заданных параметров
def get_array():
    numpy_array = data.values
    return numpy_array


lenght = get_array().size
minar = np.min(get_array())  # минимальное значение массива
maxar = np.max(get_array())  # максимальное значение массива


# среденее значение массива
armean = np.mean(get_array())
# Средне-квадратичное отклонение
std = np.std(get_array())
# доверительный интервал
dov = st.t.interval(P, len(get_array())-1, loc=armean,
                    scale=st.sem(get_array()))
# Дисперсия
dis = np.var(get_array())
# Размах массива
scope = maxar - minar
# Формула стерджесса
sterjes = 1 + 3.322 + log10(lenght)


# почти готов но ось у не в том виде
# Получение точек по кторым строиться кривая
x_axis = np.arange(minar, maxar, C)

# Построение гистограммы
plt.hist(get_array(), color='green', edgecolor='black',
         bins=7, range=[minar, maxar], density=True, )
plt.xlabel('Значения')
plt.ylabel('Вероятность')
plt.xlim(minar, maxar)
# Построение кривой
plt.plot(x_axis, norm.pdf(x_axis, armean, std-0.01), color="blue")


plt.show()
