#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:46:52 2020

@author: zyrafau
"""

import numpy as np
import random


high = 10 # [kg] najcięższy możliwy odważnik
k = 3 # liczba odważników
n = 6 # liczba rozwiązań początkowych
g = 10 # [m/s**2] przyspieszenie ziemskie
R = 4 # [m] maksymalna odległość od punktu podparcia dźwigni (warunek: k ≤ 2*R+1)
M = 100 # [Nm] moment siły


# wygeneruj przypadkowy zbiór odważników:
def GenRandWeighs(size, high, low = 0):
    return np.array([random.triangular(low,high) for i in range(size)])

# wygeneruj jedno przypadkowe rozwiązanie początkowe jako macierz (pionowy wektor poziomych par (m,r)):
GenRandSol = lambda MS, R : np.transpose(np.array([MS,random.sample(range(-R,R),len(MS))]))

# oblicz wartość funkcji celu obecnego rozwiązania:
ObjectiveFunc = lambda M, sol, g : abs(M-g*np.sum(np.multiply(sol[:,0],sol[:,1])))
# argument < 0 funkcji abs() oznaczałby, że dźwignia przechyla się na grot osi

def SortBestSol(S, M, g):
    F = [ObjectiveFunc(M, sol, g) for sol in S] # wartości funkcji celu dla S
    I = sorted(range(len(F)), key = lambda k : F[k]) # indeksy sortowania
    return F, I

# I etap (tworzenie pierwszego pokolenia rozwiązań):

MS = GenRandWeighs(k, high)
print('MS = ', MS)

S = [GenRandSol(MS, R) for i in range(n)]
for i in range(n):
    print('S[', i, '] =\n', S[i], '\n')

(F, I) = SortBestSol(S, M, g)
print('F = ', F)
print('I = ', I)

# TODO: II etap (stworzenie iteracji dla każdego następnego pokolenia rozwiązań):