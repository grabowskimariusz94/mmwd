#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sample
import numpy as np
import random


high = 10 # [kg] najcięższy możliwy odważnik
k = 4 # liczba odważników
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
    F = [ObjectiveFunc(M, sol, g) for sol in S]# wartości funkcji celu dla S
    I = sorted(range(len(F)), key = lambda k : F[k]) # indeksy sortowania
    return F, I

def Select(S,I):
    Selected = []
    for i in range(int(len(I)/2)):
        Selected.append(S[I[i]])
        print('Selected[', i, '] =\n', Selected[i], '\n')
    return Selected

def Crossing(S):
    left = random.sample(range(k), 2)
    NewGener = S
    for i in range(len(S)):
        j = i
        while j == i:
            j = random.randint(0,len(S)-1)
        child = []
        prohibited = []
        for gen in range(k):
            if gen in left:
                if S[i][gen][1] not in prohibited:
                    child.append(S[i][gen])
                    prohibited.append(S[i][gen][1])
                    good = True
                else:
                    p = 1
                    good = False
                    while not good:
                        if S[i][gen][1] + p <= R:
                            if S[i][gen][1]+p not in prohibited:
                                child.append([S[i][gen][0],S[i][gen][1] + p])
                                prohibited.append(S[i][gen][1]+p)
                                good = True
                        elif S[i][gen][1] - p >= -R:
                            if S[i][gen][1]-p not in prohibited:
                                child.append([S[i][gen][0],S[i][gen][1] - p])
                                prohibited.append(S[i][gen][1]-p)
                                good = True
                        p+=1
            else:
                if S[j][gen][1] not in prohibited:
                    child.append(S[j][gen])
                    prohibited.append(S[j][gen][1])
                    good = True
                else:
                    p = 1
                    good = False
                    while not good:
                        if S[j][gen][1] + p <= R:
                            if S[j][gen][1]+p not in prohibited:
                                child.append([S[j][gen][0],S[j][gen][1] + p])
                                prohibited.append(S[j][gen][1]+p)
                                good = True
                        elif S[j][gen][1] - p >= -R:
                            if S[j][gen][1]-p not in prohibited:
                                child.append([S[j][gen][0],S[j][gen][1] - p])
                                prohibited.append(S[j][gen][1]-p)
                                good = True
                        p+=1
        NewGener.append(np.array(child))
    return NewGener

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
Selected = Select(S,I)
NewGener = Crossing(Selected)
    
for i in range(n):
    print('NewGener[', i, '] =\n', NewGener[i], '\n')    

(F, I) = SortBestSol(NewGener, M, g)
print('F = ', F)
print('I = ', I)
               

