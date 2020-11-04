#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sample
import numpy as np
import random
from typing import List, Callable
import copy


high = 10 # [kg] najcięższy możliwy odważnik
k = 4 # liczba odważników
n = 6 # liczba rozwiązań początkowych
g = 10 # [m/s**2] przyspieszenie ziemskie
R = 4 # [m] maksymalna odległość od punktu podparcia dźwigni (warunek: k ≤ 2*R+1)
M = 100 # [Nm] moment siły


def printBeautiful(array: list, name: str, size: int) -> None:
    """
          Printuje listę w czytelniejszy sposób.

                  Parametry:
                        array (list): Lista.
                        name (str): Nazwa listy.
                        n (int): Rozmiar listy.
    """
    for i in range(size):
        print('{name}[{i}] =\n {value} \n'.format(name=name, i=i, value=array[i]))

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

def Crossing(S): # argumentem jest lista najlepszych wyników ze starej generacji
    left = random.sample(range(k), int(k/2))   #wybieranie genów które będą brane od rodzica i
    NewGener = S                         # przypisanie najlepszych wyników starej generacji do nowej
    for i in range(len(S)):       # każdy element starej generacji będzie rodzicem i raz
        j = i                   # j to drugi rodzic
        while j == i:
            j = random.randint(0,len(S)-1)
        child = []       # nowe rozwiązanie
        prohibited = []  # lista na zajęte już pozycje w nowym rozwiązaniu
        for gen in range(k): # pętla dla każdego ciężarka
            if gen in left:  # ciężarek będzie brany od rodzica i
                if S[i][gen][1] not in prohibited:  # jesli miejsce dla ciezarka nie jest zajete
                    child.append(S[i][gen])         # to dodaj ciezarek z miejscem do nowego rozwiazania
                    prohibited.append(S[i][gen][1]) # zajete miejsce dodaj do listy zajetych miejsc
                else:                           # jesli miejsce jest juz zajete
                    p = 1                       # to wybieramy miejsce najblizsze w okolicy
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
            else:                           # ciężarek będzie brany od rodzica j
                if S[j][gen][1] not in prohibited:
                    child.append(S[j][gen])
                    prohibited.append(S[j][gen][1])
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
        NewGener.append(np.array(child))  # rozszerzenie nowej generacji o nowe rozwiazanie
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

(F2, I2) = SortBestSol(NewGener, M, g)
print('F2 = ', F2)
print('I2 = ', I2)

# Nie usuwajcie
# Types
Solutions = List[List[List[float]]]


def mutate(currentSolutions: Solutions, maxDistance: int) -> Solutions:
    """
      Mutuje rozwiązania jeśli nie są one wystarczające.

              Parametry:
                    currentSolutions (Solutions): Obecne rozwiązania, które nie spełniają warunków.
                    maxDistance (Solutions): Maksymalny dystans, który pozwoli wygenerować listę dostępnych wszystkich miejsc.

              Returns:
                    mutatedCurrentSolutions (Solutions): Zmutowane rozwiązania do dalszej weryfikacji.
    """
    copiedCurrentSolutions = copy.deepcopy(currentSolutions)
    # print("Lista przed: \n", copiedCurrentSolutions)

    randomSolutionNumber: int = np.random.randint(0, len(copiedCurrentSolutions))
    # Randomowe rozwiązanie, w którym będziemy aplikować mutację
    randomSolution: List[List[float]] = copiedCurrentSolutions[randomSolutionNumber]

    # Wyciąga z rozwiązania zajęte miejsca
    findTakenPositions: Callable[[List[float]], int] = lambda weightDistance: weightDistance[1]

    allPositions: set = set(range(-maxDistance, maxDistance + 1))
    takenPositions: set = set(map(findTakenPositions, randomSolution))
    availablePositions: set = allPositions - takenPositions
    # print("Wszystkie miejsca: ", allPositions)
    # print("Zajęte miejsca: ", takenPositions)
    # print("Dostępne miejsca: ", availablePositions)

    randomAvailablePosition: int = random.choice(tuple(availablePositions))
    # print("Randomowe dostępne miejsce: ", randomAvailablePosition)

    # print("Randomowe rozwiązanie przed: \n", randomSolution)

    randomWeightDistanceNumber: int = np.random.randint(0, len(randomSolution))
    # print("Randomowa liczba wskazująca na parę (ciężar, pozycja): ", randomWeightDistanceNumber)
    randomWeightDistance = randomSolution[randomWeightDistanceNumber]

    # Mutuje 1 losowo wybraną parę
    randomWeightDistance[1] = randomAvailablePosition

    # Wstawienie rozwiązania z zmutowaną parą do kopii oryginalnej listy
    copiedCurrentSolutions[randomSolutionNumber] = randomSolution

    # print("Randomowe rozwiązanie po: \n", randomSolution)
    # print("Lista po: \n", copiedCurrentSolutions)
    return copiedCurrentSolutions


mutated = mutate(NewGener, R)

printBeautiful(mutated, "mutated", len(mutated))