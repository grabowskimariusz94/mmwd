#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sample
import numpy as np
import random
from typing import List, Dict
import copy
import matplotlib.pyplot as plt
from collections import deque
import time
import json


def loadParameters(fileName: str="parameters") -> Dict:
    """
          Ładuje parametry z pliku o formacie json.

                  Parametry:
                        fileName (str): Nazwa pliku
    """
    with open(fileName+".json") as f:
        variables = json.load(f)

    return variables


parameters = loadParameters()

high = parameters["theHeaviest"]  # [kg] najcięższy możliwy odważnik
k = parameters["numberOfWeights"]  # Liczba odważników
n = parameters["numberOfInitSolutions"]  # Liczba rozwiązań początkowych
g = parameters["g"]  # [m/s**2] przyspieszenie ziemskie
R = parameters["R"]  # [m] maksymalna odległość od punktu podparcia dźwigni (warunek: k ≤ 2*R+1)
M = parameters["M"]  # [Nm] moment siły

isStatic: bool = True  # z góry określony lub losowy przypadek
staticFileName = "caseTooLight"

w = parameters["parentsSize"] # liczba rodziców dla kolejnych generacji
m = 2

# def printBeautiful(array: list, name: str, size: int) -> None:
#     """
#           Printuje listę w czytelniejszy sposób.
#
#                   Parametry:
#                         array (list): Lista.
#                         name (str): Nazwa listy.
#                         n (int): Rozmiar listy.
#     """
#     for i in range(size):
#         print('{name}[{i}] =\n {value} \n'.format(name=name, i=i, value=array[i]))

# wygeneruj przypadkowy zbiór <size> odważników z zakresu liczb naturalnych [low,high] i zwróć posortowane:
def genRandWeighs(low=1, high: int=high, k: int=k):
    return np.array(sorted(random.choices(range(low,high),k=k)))

reps = lambda elem, MS: np.count_nonzero(MS==elem)

# wygeneruj jedno przypadkowe rozwiązanie początkowe jako wektor indeksów wektora MS
# wygenerowany wektor ma długość 2R+1
# puste miejsca na belce dźwigni są None
def genRandSol(R: int=R, k: int=k):
    if k>2*R+1:
        raise ValueError("odważniki nie mieszczą się na takiej dźwigni")
    Sol = np.full(2*R+1,None)
    Pos = random.sample(range(2*R+1),random.randint(1,k))
    for i in range(len(Pos)):
        Sol[Pos[i]] = i
    return Sol

# oblicz wartość funkcji celu obecnego rozwiązania:
def objectiveFunc(Sol, MS, M: float=M, g: float=g, R: int=R) -> int:
    mr = 0 # [kg·m]
    temp_sol = Sol[:len(Sol)//2]+[0]+Sol[len(Sol)//2:] if len(Sol)/2==len(Sol)//2 else Sol
    if None in temp_sol:
        for i in range(len(temp_sol)):
            if temp_sol[i] is not None:
                mr += MS[temp_sol[i]]*(i-R)
    else:
        for i in range(len(temp_sol)):
            if temp_sol[i] != 0:
                mr += temp_sol[i]*(i-R)
    return abs(M-g*mr)
# argument < 0 funkcji abs() oznaczałby, że dźwignia przechyla się na grot osi

def sortBestSol(S, MS, M: float=M, g: float=g, R: int=R):
    F = [objectiveFunc(Sol, MS, M, g, R) for Sol in S] # wartości funkcji celu dla S
    Idx = sorted(range(len(F)), key = lambda k : F[k]) # indeksy sortowania
    return F, Idx

def transformSol(S, MS):
    f = lambda x : 0 if x is None else MS[x]
    g = lambda x : list(map(f,x))
    return list(map(g,S))

def Select(S,I):
    if len(I) < w:
        return S;
    Selected = []
    ''' 
    # mocno elitarnie
    not_best_selected = random.sample(range(n-int(n/3)), int(n/2)-int(n/3))
    not_best_selected=[x+int(n/3) for x in not_best_selected]
    for i in range(int(n/2)):
        if(i<int(n/3)):
            Selected.append(S[I[i]])
        else:
            Selected.append(S[I[not_best_selected[i-int(n/3)]]])
        # print('Selected[', i, '] =\n', Selected[i], '\n')
    '''
    # mniej elitarnie
    prohibited = []
    for i in range(w):
        good = False
        while good!=True:
            random_selected = random.sample(range(len(I)), m)
            for i in range(len(random_selected)):
                min_random_selected = min(random_selected)
                if (min_random_selected  not in prohibited ):
                    prohibited.append(min_random_selected)
                    good = True
                    break;
                else:
                    random_selected.remove(min_random_selected)
            
        Selected.append(S[I[min_random_selected]])
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
                x = S[i][gen]
            else: 
                x = S[j][gen]
            if(x!=0):
                # while (child.count(x)>=np.count_nonzero(MS == x)):    #gdy brakuje ciężarków
                #     x = random.choice(MS)               #to wylosuj na to miejsce inny
                s = 0
                while (child.count(x) >= np.count_nonzero(MS == x)):  # gdy brakuje ciężarków
                    s += 1
                    x = random.choice(MS)  # to wylosuj na to miejsce inny
                    if s == 10:
                        break
            child.append(x)         # dodaj ciezarek z miejscem do nowego rozwiazania
        NewGener.append(child)  # rozszerzenie nowej generacji o nowe rozwiazanie
   return NewGener

def sick(kid, MS) -> bool:
    Traits = set(kid)
    if 0 in Traits:
        Traits.remove(0)
    for trait in Traits:
        if kid.count(trait)>reps(trait,MS):
            return True
    return False

# Types
Solutions = List[List[float]]

def alternativeCrossing(S, MS, w) -> Solutions: # zabija rodziców, zatem zmniejsza zbieżność
    _, Idx = sortBestSol(S, MS)
    Daddies = random.sample(range(w),random.randint(1,w//2))
    sortDaddies = sorted(range(len(Daddies)), key = lambda k : Idx[Daddies[k]])
    Daddies = [Daddies[idx] for idx in sortDaddies]
    sortMummies = sorted(range(len(Idx)), key = lambda k : Idx[k])
    Mummies = [mummy for mummy in sortMummies if mummy not in Daddies]
    NewGen = []
    iterations = len(Daddies) if len(Daddies)<len(Mummies) else len(Mummies)
    while(True):
        for i in range(iterations):
            if len(NewGen)==w:
                return NewGen
            kid = [random.choice([S[Daddies[i]][pos],S[Mummies[i]][pos]]) for pos in range(len(S[0]))]
            if not sick(kid, MS): # kid not in NewGen and … (żeby się nie powtarzały)
                NewGen.append(kid)


def rotateMutation(currentSolutions: Solutions, offset: int=1) -> Solutions:
    """
       Rotuje rozwiązania jeśli nie są one wystarczające.

               Parametry:
                     currentSolutions (Solutions): Obecne rozwiązania, które nie spełniają warunków
                     offset (Solutions): Przesunięcie

               Returns:
                     copiedCurrentSolutions (Solutions): Zmutowane rozwiązania do dalszej weryfikacji
     """
    copiedCurrentSolutions = copy.deepcopy(currentSolutions)

    randomSolutionNumber: int = np.random.randint(0, len(copiedCurrentSolutions))

    # Randomowe rozwiązanie, w którym będziemy aplikować mutację
    randomSolution = deque(copiedCurrentSolutions[randomSolutionNumber])
    randomSolution.rotate(offset)

    copiedCurrentSolutions[randomSolutionNumber] = list(randomSolution)

    return copiedCurrentSolutions


def mutate(currentSolutions: Solutions, maxDistance: int, log: bool=False) -> Solutions:
    """
      Mutuje rozwiązania jeśli nie są one wystarczające.

              Parametry:
                    currentSolutions (Solutions): Obecne rozwiązania, które nie spełniają warunków
                    maxDistance (Solutions): Maksymalny dystans, który pozwoli wygenerować listę dostępnych wszystkich miejsc
                    log (bool): Określa pokazywanie informacji procesu

              Returns:
                    copiedCurrentSolutions (Solutions): Zmutowane rozwiązania do dalszej weryfikacji
    """
    copiedCurrentSolutions = copy.deepcopy(currentSolutions)
    randomSolutionNumber: int = np.random.randint(0, len(copiedCurrentSolutions))

    # Randomowe rozwiązanie, w którym będziemy aplikować mutację
    randomSolution: List[float] = copiedCurrentSolutions[randomSolutionNumber]

    allPositions: set = set(range(2*maxDistance + 1))
    # Wyciąga z rozwiązania zajęte miejsca
    takenPositions: set = set([index for index, weight in enumerate(randomSolution) if weight])
    availablePositions: set = allPositions - takenPositions
    if not len(availablePositions):
        randomFirstPos: int = random.choice(tuple(takenPositions))
        randomSecondPos: int = random.choice(tuple(takenPositions))
        randomSolution[randomFirstPos], randomSolution[randomSecondPos] = randomSolution[randomSecondPos], randomSolution[randomFirstPos]
        copiedCurrentSolutions[randomSolutionNumber] = randomSolution
        return copiedCurrentSolutions
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
                      currentSolutions (Solutions): Obecne rozwiązania
                      mutatedSolutions (Solutions): Zmutowane rozwiązania
                      torque (float): Moment siły
                      gravity (float): Przyspieszenie

                Returns:
                      (bool): Wiadomość o korzystniejszym stanie
    """
    def objectiveFunction(solutions: Solutions) -> float:
        """
             Zwraca najlepszy rezultat w danych rozwiązaniach.

                     Parametry:
                           solutions (Solutions): Rozwiązania

                     Returns:
                           bestResult (float): Najlepszy rezultat
         """
        bestResult = min([abs(torque - gravity *
                              sum([weight * (index - maxDistance) for index, weight in enumerate(solution) if weight]))
                          for solution in solutions])

        return bestResult

    bestCurrentResult: float = objectiveFunction(currentSolutions)
    bestMutatedResult: float = objectiveFunction(mutatedSolutions)
    print("Obecne najlepsze: ", bestCurrentResult, "Zmutowane najlepsze: ", bestMutatedResult)

    return bestMutatedResult >= bestCurrentResult

def cannotSolve(MS, M: float=M, g: float=g, R: int=R) -> bool:
    mr = 0
    review = abs(M)/g
    for r in range(R):
        if r<len(MS):
            mr += MS[-(r+1)]*(R-r)
        else:
            return True
        if mr>=review:
            return False
    return True
    

# I etap (tworzenie pierwszego pokolenia rozwiązań):
timestamps = [time.clock()]
# 1. sposób:
#MS = genRandWeighs()

# 2. sposób:
MS = 5*[8]+4*[6]+3*[5]+4*[4]+5*[3]+3*[2]
staticParameters: Dict = dict()
if isStatic:
    staticParameters = loadParameters("caseOneSided")

    staticCollection: List = list()
    for collection in staticParameters["collectionOfWeights"]:
        weight, amount = collection.values()
        staticCollection += amount*[weight]

    # Override parameters - static case
    MS = staticCollection
    k = len(MS)
    M = staticParameters["M"]
    R = staticParameters["R"]
    w = staticParameters["parentsSize"]

print(len(MS), k)
if len(MS)!=k:
    raise ValueError("parametr k nie zgadza się z wektorem odważników MS")
MS = np.array(sorted(MS))
if cannotSolve(MS, M, g, R):
    raise ValueError("odważniki są zbyt lekkie, żeby zredukować wypadkowy moment siły do zera")

print('MS =', MS)
print('Powtórzenia \'2\' w MS:', sum(MS==2), '\n') # powtórzenia w np.array

# Override parameters - static case
staticSolutions = staticParameters.get("solutions")
if staticSolutions:
    n = len(staticSolutions)

S = [genRandSol() for i in range(n)]
if staticSolutions:
    S = staticSolutions
if not isStatic:
    S = transformSol(S, MS) # zakomentuj to, jeśli chcesz działać na indeksach a nie na masach
for i in range(n):
    print('S[', i, '] =\n', S[i])
    print('Powtórzenia \'3\' w S[', i, ']:', S[i].count(3), '\n') # powtórzenia w liście

# Odkomentuj jeśli chcesz zobaczyć mutate
# mutate(S, R, True)
# print(markMutation(S, S, M, g, R))

F, Idx = sortBestSol(S, MS)

timestamps.append(time.clock())

print('F = ', F)

print('Idx = ', Idx)

#S = alternativeCrossing(S, MS, w)
#for i in range(n):
#    print('S[', i, '] =\n', S[i])
#
#F, Idx = sortBestSol(S, MS)
#print('F = ', F)
#
#print('Idx = ', Idx)

# II etap (stworzenie iteracji dla każdego następnego pokolenia rozwiązań):
Selected = Select(S,Idx)
for i in range(w):
    print('Selected[', i, '] =\n', Selected[i], '\n')

#for i in range(n):
#   print('NewGener[', i, '] =\n', NewGener[i], '\n')    
bestchild = []
bestchild.append(F[Idx[0]])

best = S[Idx[0]]
best_value = F[Idx[0]]
champion = [F[Idx[0]]] # historycznie najlepsze rozwiązanie

# ---------- Calculation settings ----------
howOftenMutation = parameters["howOftenMutation"]  # Co jaki czas ma się pojawiać próba mutacji
amountMutationAttempts = parameters["amountMutationAttempts"]  # Ilość mutacji w danej próbie
generations = parameters["generations"]-1  # Liczba generacji wtórnych (minus 1 iteracja wstępna)
alternativeCrossingFrequency = parameters["alternativeCrossingFrequency"]  # Częstotliwość usuwania rozwiązań macierzystych
alternativeMutationFrequency = parameters["alternativeMutationFrequency"]
# ---------- End  ----------
# ---------- Private calculation settings ----------
counterAlternativeMutation = 0  # do not change!
# ---------- End  ----------

timestamps.append(time.clock())

for i in range(generations):
    if not best_value:
        break
    mutated = None
    newGener = None
    if i%alternativeCrossingFrequency:
        NewGener = Crossing(Selected)
    else:
        NewGener = alternativeCrossing(Selected, MS, w)

    if i==0:
        for j in range(int(len(NewGener))):
            print('NewGener[', j, '] =\n', NewGener[j], '\n')
    mutationFlag = True

    if not (i+1) % howOftenMutation:
        mutated = mutate(NewGener, R)
        # mutationFlag: bool = markMutation(NewGener, mutated, M, g, R)
        # if mutationFlag:
        #     counter = 0
        #     copyMutated = copy.deepcopy(mutated)
        #     attempts = amountMutationAttempts
        #     while mutationFlag and counter <= attempts:
        #         copyMutated = mutate(NewGener, R)
        #         mutationFlag = markMutation(NewGener, copyMutated, M, g, R)
        #         counter += 1
        #     if counter <= attempts:
        #         mutated = copyMutated

    (F, Idx) = sortBestSol(NewGener if not mutated else mutated, MS, M)
    bestchild.append(F[Idx[0]])
    if(F[Idx[0]]<best_value):
        best_value = F[Idx[0]]
        best = NewGener[Idx[0]] if mutationFlag else mutated[Idx[0]]
        counterAlternativeMutation = 0
    else:
        counterAlternativeMutation += 1
    # print(counterAlternativeMutation)
    if counterAlternativeMutation == alternativeMutationFrequency:
        #randomOffset: int = np.random.randint(0, 2*R)
        mutated = rotateMutation(NewGener if not mutated else mutated)
        counterAlternativeMutation = 0

    champion.append(best_value)
    Selected = Select(NewGener if not mutated else mutated, Idx)
    # (F, I) = sortBestSol(NewGener if mutationFlag else mutated, M, g)
    # bestchild.append(F[I[0]])
    # Selected = Select(NewGener if mutationFlag else mutated, I)
    #plt.plot(best_value)

    # print(F[I[0]])
    timestamps.append(time.clock())

durations = [1000*(timestamps[idx+1]-timestamps[idx]) for idx in range(len(timestamps)-1)]
durations.pop(1)
#print(sum(durations)) # czas algorytmu z pominięciem operacji pomiędzy iteracjami
plt.figure()
plt.title('Algorithm\'s duration: '+str(1000*(timestamps[-1]-timestamps[0]))+' ms')
plt.plot(durations)
plt.axvline(x=0,linewidth='1',linestyle=':',c='m')
plt.axvline(x=0,linewidth='1',linestyle=':',c='y')
plt.xlabel('generation')
plt.ylabel('time [ms]')
for everyMutation in range(1,generations//howOftenMutation+1):
    plt.axvline(x=howOftenMutation*everyMutation,linewidth='1',linestyle=':',c='m')
for everyAlternativeCrossing in range(1,generations//alternativeCrossingFrequency+1):
    plt.axvline(x=alternativeCrossingFrequency*everyAlternativeCrossing,linewidth='1',linestyle=':',c='y')
plt.legend(['durations','mutations','alternative crossings'],loc='upper right')
plt.show()

print(bestchild)
plt.figure()
plt.plot(bestchild)
plt.plot(champion[1:])
plt.axvline(x=0,linewidth='1',linestyle=':',c='m')
plt.axvline(x=0,linewidth='1',linestyle=':',c='y')
#plt.grid()
plt.title('Evolutionary algorithm')
plt.xlabel('generation')
plt.ylabel('objective function')
for everyMutation in range(1,generations//howOftenMutation+1):
    plt.axvline(x=howOftenMutation*everyMutation,linewidth='1',linestyle=':',c='m')
for everyAlternativeCrossing in range(1,generations//alternativeCrossingFrequency+1):
    plt.axvline(x=alternativeCrossingFrequency*everyAlternativeCrossing,linewidth='1',linestyle=':',c='y')
plt.legend(['bestchild','champion','mutations','alternative crossings'],loc='upper right')
plt.show()
print(best)
print(best_value)

#printBeautiful(mutated, "mutated", len(mutated))


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
