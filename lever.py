#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sample
import numpy as np
import random
from typing import List, Callable
import copy
import matplotlib.pyplot as plt


high = 100 # [kg] najcięższy możliwy odważnik
k = 9 # liczba odważników
n = 6 # liczba rozwiązań początkowych
g = 10 # [m/s**2] przyspieszenie ziemskie
R = 8 # [m] maksymalna odległość od punktu podparcia dźwigni (warunek: k ≤ 2*R+1)
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

# wygeneruj przypadkowy zbiór <size> odważników z zakresu liczb naturalnych [low,high] i zwróć posortowane:
def genRandWeighs(low = 1, high: int = high, k: int = k):
    return np.array(sorted(random.choices(range(low,high),k = k)))

# wygeneruj jedno przypadkowe rozwiązanie początkowe jako wektor indeksów wektora MS
# wygenerowany wektor ma długość 2R+1
# puste miejsca na belce dźwigni są None
def genRandSol(R: int = R, k: int = k):
    if k>2*R+1:
        raise ValueError("odważniki nie mieszczą się na takiej dźwigni")
    Sol = np.full(2*R+1,None)
    Pos = random.sample(range(2*R+1),k)
    for i in range(len(Pos)):
        Sol[Pos[i]] = i
    return Sol

# oblicz wartość funkcji celu obecnego rozwiązania:
def objectiveFunc(Sol, MS, M: float, g: float = 10) -> int:
    mr = 0 # [kg·m]
    if None in Sol:
        for i in range(len(Sol)):
            if Sol[i] is not None:
                mr += MS[Sol[i]]*(i-R)
    else:
        for i in range(len(Sol)):
            if Sol[i] != 0:
                mr += Sol[i]*(i-R)
    return abs(M-g*mr)
# argument < 0 funkcji abs() oznaczałby, że dźwignia przechyla się na grot osi

def sortBestSol(S, MS, M: float, g: float = 10):
    F = [objectiveFunc(Sol, MS, M, g) for Sol in S] # wartości funkcji celu dla S
    Idx = sorted(range(len(F)), key = lambda k : F[k]) # indeksy sortowania
    return F, Idx

def transformSol(S, MS):
    f = lambda x : 0 if x is None else MS[x]
    g = lambda x : list(map(f,x))
    return list(map(g,S))

def Select(S,I):
    Selected = []
    not_best_selected = random.sample(range(n-int(n/3)), int(n/2)-int(n/3))
    not_best_selected=[x+int(n/3) for x in not_best_selected]
    for i in range(int(n/2)):
        if(i<int(n/3)):
            Selected.append(S[I[i]])
        else:
            Selected.append(S[I[not_best_selected[i-int(n/3)]]])
        # print('Selected[', i, '] =\n', Selected[i], '\n')
    return Selected

def Crossing(S):                                # argumentem jest lista najlepszych wyników ze starej generacji
   left = random.sample(range(2*R+1), int(R+1))   #wybieranie miejsc które będą brane od rodzica i
   NewGener = copy.deepcopy(S)                         # przypisanie najlepszych wyników starej generacji do nowej
   for i in range(len(S)):       # każdy element starej generacji będzie rodzicem i raz
        j = i                   # j to drugi rodzic
        while j == i:
            j = random.randint(0,len(S)-1)
        child = []       # nowe rozwiązani
        for gen in range(2*R+1): # pętla dla każdego miejsca
            if gen in left:  # ciężarek będzie brany od rodzic
                child.append(S[i][gen])         # to dodaj ciezarek z miejscem do nowego rozwiazania
            else : 
                child.append(S[j][gen])
        NewGener.append(child)  # rozszerzenie nowej generacji o nowe rozwiazanie
   return NewGener

#def Crossing(S): # argumentem jest lista najlepszych wyników ze starej generacji
#   left = random.sample(range(k), int(k/2))   #wybieranie genów które będą brane od rodzica i
#    NewGener = copy.deepcopy(S)                         # przypisanie najlepszych wyników starej generacji do nowej
#    for i in range(len(S)):       # każdy element starej generacji będzie rodzicem i raz
#        j = i                   # j to drugi rodzic
#        while j == i:
#            j = random.randint(0,len(S)-1)
#        child = []       # nowe rozwiązanie
#        prohibited = []  # lista na zajęte już pozycje w nowym rozwiązaniu
#        for gen in range(k): # pętla dla każdego ciężarka
#            if gen in left:  # ciężarek będzie brany od rodzica i
#                if S[i][gen][1] not in prohibited:  # jesli miejsce dla ciezarka nie jest zajete
#                    child.append(S[i][gen])         # to dodaj ciezarek z miejscem do nowego rozwiazania
#                    prohibited.append(S[i][gen][1]) # zajete miejsce dodaj do listy zajetych miejsc
#                else:                           # jesli miejsce jest juz zajete
#                    p = 1                       # to wybieramy miejsce najblizsze w okolicy
#                    good = False
#               while not good:
#                       if S[i][gen][1] + p <= R:
#                            if S[i][gen][1]+p not in prohibited:
#                                child.append(np.array([S[i][gen][0],S[i][gen][1] + p]))
#                                prohibited.append(S[i][gen][1]+p)
#                                good = True
#                        elif S[i][gen][1] - p >= -R:
#                            if S[i][gen][1]-p not in prohibited:
#                                child.append(np.array([S[i][gen][0],S[i][gen][1] - p]))
#                                prohibited.append(S[i][gen][1]-p)
#                                good = True
#                        p+=1
#            else:                           # ciężarek będzie brany od rodzica j
#                if S[j][gen][1] not in prohibited:
#                    child.append(S[j][gen])
#                    prohibited.append(S[j][gen][1])
#                else:
#                    p = 1
#                    good = False
#                    while not good:
#                        if S[j][gen][1] + p <= R:
#                            if S[j][gen][1]+p not in prohibited:
#                                child.append(np.array([S[j][gen][0],S[j][gen][1] + p]))
#                                prohibited.append(S[j][gen][1]+p)
#                                good = True
#                        elif S[j][gen][1] - p >= -R:
#                            if S[j][gen][1]-p not in prohibited:
#                                child.append(np.array([S[j][gen][0],S[j][gen][1] - p]))
#                                prohibited.append(S[j][gen][1]-p)
#                                good = True
#                        p+=1
#        NewGener.append(np.array(child))  # rozszerzenie nowej generacji o nowe rozwiazanie
#    return NewGener

# Nie usuwajcie
# Types
Solutions = List[List[float]]

def mutate(currentSolutions: Solutions, maxDistance: int, log: bool = False) -> Solutions:
    """
      Mutuje rozwiązania jeśli nie są one wystarczające.

              Parametry:
                    currentSolutions (Solutions): Obecne rozwiązania, które nie spełniają warunków.
                    maxDistance (Solutions): Maksymalny dystans, który pozwoli wygenerować listę dostępnych wszystkich miejsc.
                    log (bool): Określa pokazywanie informacji procesu.

              Returns:
                    copiedCurrentSolutions (Solutions): Zmutowane rozwiązania do dalszej weryfikacji.
    """
    copiedCurrentSolutions = copy.deepcopy(currentSolutions)

    randomSolutionNumber: int = np.random.randint(0, len(copiedCurrentSolutions))
    # Randomowe rozwiązanie, w którym będziemy aplikować mutację
    randomSolution: List[float] = copiedCurrentSolutions[randomSolutionNumber]

    allPositions: set = set(range(2*maxDistance + 1))
    # Wyciąga z rozwiązania zajęte miejsca
    takenPositions: set = set([index for index, weight in enumerate(randomSolution) if weight])
    availablePositions: set = allPositions - takenPositions
    if log:
        print("\n\nMutate...")
        print("Wszystkie miejsca: ", allPositions)
        print("Zajęte miejsca: ", takenPositions)
        print("Dostępne miejsca: ", availablePositions)

    randomAvailablePosition: int = random.choice(tuple(availablePositions))
    randomWeightDistanceNumber: int = random.choice(tuple(takenPositions))
    if log:
        print("Randomowe dostępne miejsce: ", randomAvailablePosition)
        print("Randomowe zajęte miejsce: ", randomWeightDistanceNumber)
        print("Randomowe rozwiązanie przed: \n", randomSolution)

    # Mutuje 1 losowo wybraną parę
    randomSolution[randomAvailablePosition] = randomSolution[randomWeightDistanceNumber]
    randomSolution[randomWeightDistanceNumber] = 0
    if log:
        print("Randomowe rozwiązanie po: \n", randomSolution)
        print("End mutate\n\n")

    # Wstawienie rozwiązania z zmutowaną parą do kopii oryginalnej listy
    copiedCurrentSolutions[randomSolutionNumber] = randomSolution

    # print("Lista po: \n", copiedCurrentSolutions)
    return copiedCurrentSolutions

def markMutation(currentSolutions: Solutions, mutatedSolutions: Solutions, torque: float, gravity: float, maxDistance: int) -> bool:
    """
        Ocenia mutację pod względem lepszego rezultatu. Jeśli wynik jest korzystniejszy zwraca False.

                Parametry:
                      currentSolutions (Solutions): Obecne rozwiązania.
                      mutatedSolutions (Solutions): Zmutowane rozwiązania.
                      torque (float): Moment siły.
                      gravity (float): Przyspieszenie.

                Returns:
                      (bool): Wiadomość o korzystniejszym stanie.
    """
    def objectiveFunction(solutions: Solutions) -> float:
        """
             Zwraca najlepszy rezultat w danych rozwiązaniach.

                     Parametry:
                           solutions (Solutions): Rozwiązania.

                     Returns:
                           bestResult (float): Najlepszy rezultat.
         """
        bestResult = min([abs(torque - gravity *
                              sum([weight * (index - maxDistance) for index, weight in enumerate(solution) if weight]))
                          for solution in solutions])

        return bestResult

    bestCurrentResult: float = objectiveFunction(currentSolutions)
    bestMutatedResult: float = objectiveFunction(mutatedSolutions)
    print("Obecne najlepsze: ", bestCurrentResult, "Zmutowane najlepsze: ", bestMutatedResult)

    return bestMutatedResult >= bestCurrentResult

# I etap (tworzenie pierwszego pokolenia rozwiązań):

MS = genRandWeighs()
print('MS = ', MS)

S = [genRandSol() for i in range(n)]
S = transformSol(S, MS) # zakomentuj to, jeśli chcesz działać na indeksach a nie na masach
for i in range(n):
    print('S[', i, '] =\n', S[i], '\n')

# Odkomentuj jeśli chcesz zobaczyć mutate
# mutate(S, R, True)
# print(markMutation(S, S, M, g, R))

(F, Idx) = sortBestSol(S, MS, M)
print('F = ', F)

print('Idx = ', Idx)


# II etap (stworzenie iteracji dla każdego następnego pokolenia rozwiązań):
Selected = Select(S,Idx)
for i in range(int(n/2)):
    print('Selected[', i, '] =\n', Selected[i], '\n')

#for i in range(n):
#   print('NewGener[', i, '] =\n', NewGener[i], '\n')    
bestchild = []
bestchild.append(F[Idx[0]])

# ---------- Calculation settings ----------
howOftenMutation = 200  # Co jaki czas ma się pojawiać próba mutacji
amountMutationAttempts = 100  # Ilość mutacji w danej próbie
generations = 1000  # Liczba generacji
# ---------- End  ----------

for i in range(generations):
    NewGener = Crossing(Selected)
    if i==0:
        for j in range(int(len(NewGener))):
            print('NewGener[', j, '] =\n', NewGener[j], '\n')
    mutationFlag = True
    mutated = None
    if not (i+1) % howOftenMutation:
        mutated = mutate(NewGener, R)
        mutationFlag: bool = markMutation(NewGener, mutated, M, g, R)
        if mutationFlag:
            counter = 0
            copyMutated = copy.deepcopy(mutated)
            attempts = amountMutationAttempts
            while mutationFlag and counter <= attempts:
                copyMutated = mutate(NewGener, R)
                mutationFlag = markMutation(NewGener, copyMutated, M, g, R)
                counter += 1
            if counter <= attempts:
                mutated = copyMutated

    (F, Idx) = sortBestSol(NewGener if mutationFlag else mutated, MS, M)
    bestchild.append(F[Idx[0]])
    Selected = Select(NewGener if mutationFlag else mutated, Idx)
    # (F, I) = sortBestSol(NewGener if mutationFlag else mutated, M, g)
    # bestchild.append(F[I[0]])
    # Selected = Select(NewGener if mutationFlag else mutated, I)


    # print(F[I[0]])
print(bestchild)
plt.plot(bestchild)
plt.show()

#printBeautiful(mutated, "mutated", len(mutated))

