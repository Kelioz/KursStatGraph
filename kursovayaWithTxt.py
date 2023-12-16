import numpy as np
from math import *
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

P = 0.92  # Доверительная вероятность
q = 0.05  # Уровень значимости
C = 0.01  # Цена деления прибора С, мм

path = 'qwe.txt'
# Изменение с запятой на точку для преобразования str в float
with open(path, 'r') as fa:
    old_data = fa.read()
new_data = old_data.replace(',', '.')
with open(path, 'w') as fa:
    fa.write(new_data)


# создание массива из заданных параметров
def Array():
    a = np.loadtxt(path, usecols=1, dtype='float')
    b = np.loadtxt(path, usecols=3, dtype='float')
    c = np.loadtxt(path, usecols=5, dtype='float')
    d = np.loadtxt(path, usecols=7, dtype='float')
    e = np.loadtxt(path, usecols=9, dtype='float')
    f = np.loadtxt(path, usecols=-1, dtype='float')
    f = np.delete(f, 8)
    q = np.concatenate((a, b, c, d, e, f))
    return q


lenght = Array().size
minarray = np.min(Array())  # минимальное значение массива
maxarray = np.max(Array())  # максимальное значение массива


# среденее значение массива
def Get_mean():
    araymean = np.mean(Array(), dtype=float)
    return araymean


# Средне-квадратичное отклонение
def Get_std():
    std = np.std(Array())
    return std


# доверительный интервал
def Conf_interval():
    conf = st.t.interval(P, len(Array())-1, loc=Get_mean(),
                         scale=st.sem(Array()))
    return conf


# Дисперсия
def Disperssion():
    dis = np.var(Array())
    return dis


# Размах массива
def Scopearray():
    scope = maxarray - minarray
    return scope


# Формула стерджесса
def get_bins():
    b = 1 + 3.322 + log10(lenght)
    return b


print("размах массива", Scopearray())
print("Дисперсия", Disperssion())
print("доверительный интервал", Conf_interval())
print("Средне-квадратичное отклонение", Get_std())
print("среденее значение массива", Get_mean())


# создание гистограммы и распределения через seaburn
def seagraph():
    graph = sns.displot(data=Array(), kde=True, bins=ceil(get_bins()),)

    graph.set_xlabels('Значение')
    graph.set_ylabels('Вероятность')
    return


# почти готов но ось у не в том виде
def matbar():
    # Получение точек по кторым строиться кривая
    x_axis = np.arange(minarray, maxarray, C)

    # Построение гистограммы
    plt.hist(Array(), color='blue', edgecolor='black',
             bins=int(ceil(get_bins())), range=[minarray, maxarray], density=True, )

    # Создание до значений по X
    plt.xticks(Array())
    # Уменьшение размера значений по X
    plt.tick_params(axis='x', labelsize=8)

    plt.ylim(0, 12)
    plt.xlabel('Значения')
    plt.ylabel('Вероятность')
    plt.xlim(minarray, maxarray)

    # plt.text(
    #   42.163, 11.5, f"Значения округлены:", fontsize=8)
    plt.text(
        42.163, 10.9+0.1, f"Размах массива {round (Scopearray(), 2)}", fontsize=8, bbox=dict(facecolor="none", edgecolor="black", pad=3))
    plt.text(
        42.163, 9.9+0.4, f"Дисперсия  {round(Disperssion(), 6)}", fontsize=8, bbox=dict(facecolor="none", edgecolor="black", pad=3))
    plt.text(
        42.163, 8.9+0.7, f"Д.интервал 42.08, 42.11", fontsize=7, bbox=dict(facecolor="none", edgecolor="black", pad=3))
    plt.text(
        42.163, 7.9+1, f"С-К.отклонение 0.05299", fontsize=7, bbox=dict(facecolor="none", edgecolor="black", pad=3))
    plt.text(
        42.163, 6.9+1.3, f"Срзнач. массив {round(Get_mean(), 4)}", fontsize=7, bbox=dict(facecolor="none", edgecolor="black", pad=3))

    # Построение кривой
    plt.plot(x_axis, norm.pdf(x_axis, Get_mean(), Get_std()-0.01), color="red")

    return


matbar()

plt.show()
