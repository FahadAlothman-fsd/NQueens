# n_queens.py

"""
N-Queens Problem using Codeskulptor/SimpleGUICS2Pygame
By Robin Andrews - info@compucademy.co.uk
https://compucademy.net/blog/
"""


from copy import deepcopy
import numpy as np
import time
from queue import PriorityQueue
import random
import math


try:
    import simplegui

    SIMPLEGUICS2PYGAME = False

    # collision_sound = simplegui.load_sound("https://compucademy.net/assets/buzz3x.mp3")
    success_sound = simplegui.load_sound(
        "https://compucademy.net/assets/treasure-found.mp3")

except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
    SIMPLEGUICS2PYGAME = True
    simplegui.Frame._hide_status = True
    simplegui.Frame._keep_timers = False
    # collision_sound = simplegui.load_sound("https://compucademy.net/assets/buzz3x.wav")
    success_sound = simplegui.load_sound(
        "https://compucademy.net/assets/treasure-found.wav")

if SIMPLEGUICS2PYGAME:
    from sys import version as python_version
    from pygame.version import ver as pygame_version
    from SimpleGUICS2Pygame import _VERSION as GUI_VERSION  # pylint: disable=ungrouped-imports  # noqa

    PYTHON_VERSION = 'Python ' + python_version.split()[0]
    PYGAME_VERSION = 'Pygame ' + pygame_version
    GUI_VERSION = 'SimpleGUICS2Pygame ' + GUI_VERSION
else:
    PYTHON_VERSION = 'CodeSkulptor'  # http://www.codeskulptor.org/ or https://py3.codeskulptor.org/  # noqa
    PYGAME_VERSION = ''
    GUI_VERSION = 'simplegui'
QUEEN = 1
EMPTY_SPOT = 0
BOARD_SIZE = 8


class AstarSearch:
    global pq
    pq = PriorityQueue()
    global pq_list
    pq_list = []

    class Node:
        def __init__(self, N=0):

            self.state = np.zeros(N, dtype=int)
            self.g_x = 0
            self.h_x = 0
            self.cost_so_far = 0
            self.parent = None

    def is_goal(self, state):

        if self.cal_heuristic(state) != 0:
            return 0
        return 1

    def random_state(n):

        board = random.sample(range(n), n)

        return board

    def cal_heuristic(state):

        score = (len(state) - 1) * len(state)
        idealScore = (len(state) - 1) * len(state) / 2
        for i in range(0, len(state)):
            for j in range(0, len(state)):
                if i != j:
                    if i - state[i] == j - state[j] or i + state[i] == j + state[j]:
                        score -= 1
        score = score / 2
        return abs(score - idealScore)

    def cal_g(state1, state2):

        if (np.array_equal(state1, state2) == 0):
            changed_state = np.absolute(state1 - state2)
            ones = np.where(changed_state != 0)

            return len(ones)
        else:
            return 10

    def mutate(solution):

        newsolution = solution[:]
        i = 0
        j = 0
        while i == j:
            i = random.randint(0, len(solution) - 1)
            j = random.randint(0, len(solution) - 1)
        temp1 = newsolution[j]
        temp2 = newsolution[i]
        newsolution[i] = temp1
        newsolution[j] = temp2
        return newsolution

    def populate(self, x):
        global pq
        global pq_list
        a = None
        temp = None
        state = np.copy(x.state)

        for col in range(len(state)):
            state = np.copy(x.state)

            for row in range(len(state)):
                temp = None
                temp = self.Node()
                temp.parent = x

                temp.state = self.mutate(state)

                temp.h_x = self.cal_heuristic(temp.state)
                temp.g_x = self.cal_g(temp.state, x.state)
                temp.cost_so_far = temp.h_x + temp.g_x + x.g_x
                if np.array_equal(x.state, temp.state) == 0:
                    a = deepcopy(temp)
                    pq_list.append(a)

                    pq.put((temp.cost_so_far, len(pq_list) - 1))

    @staticmethod
    def run_AstarSearch(self, n):
        global pq
        global pq_list
        N = n
        start_state = self.random_state(N)
        start = self.Node()
        start.state = start_state
        while(True):
            self.populate(AstarSearch, start)
            next_indice = pq.get()
            next_state = pq_list[next_indice[1]]
            if(self.is_goal(AstarSearch, next_state.state) == 1):
                print("Number of Nodes expanded", len(pq_list))
                NodesExpanded = len(pq_list)
                pq_list = []
                pq.queue.clear()
                return (next_state.state, NodesExpanded)
                break

            start = next_state


class GeneticAlgorithm:
    # Global Variables
    crossover_rate = 1
    mutatation_rate = 1
    Elitism = 1
    crossover_type = 0

    def create_population(number_of_solutions, n):  # create inital population
        # N is the number of queens
        population = []
        for i in range(number_of_solutions):
            population.append(random.sample(range(n), n))
        return population

    def create_children(self, parents):
        myChildren = []
        population_size = len(parents)
        fitness_parents = []

        fixedPop_flag = True
        fitness_parents = [self.getFitness(each_sol) for each_sol in parents]

        if self.Elitism == 1:  # if there is elitism we only create n-2 children and pass the most two maximum fitness to the next generations
            population_size -= 2
            # print("in elitism")
        while len(myChildren) <= population_size - 1:
            # Certain  number  of chromosomes will  pass  onto  the next  generation depending  on  a  selection  operator
            probability_matrix = [x / sum(fitness_parents)
                                  for x in fitness_parents]
            # two parents will be selected based on their probability which is based on their fitness
            ns = np.random.choice([i for i in range(len(parents))],
                                  size=2, p=probability_matrix)
            # the selected parents are then passed to their children either by crossover or just mutation
            selectedParents = [parents[i] for i in ns]
            if self.probabalistic_decision(self.crossover_rate):
                myChildren.extend(self.create_offspring(
                    GeneticAlgorithm, selectedParents))
            else:
                myChildren.extend(self.mutate(
                    GeneticAlgorithm, selectedParents))
        # passing down the most two fit members of the next generation after being mutated
        if self.Elitism == 1:
            myChildren.append(self.mutate(GeneticAlgorithm, parents.pop(
                fitness_parents.index(max(fitness_parents)))))
            fitness_parents.pop(fitness_parents.index(max(fitness_parents)))
            myChildren.append(self.mutate(GeneticAlgorithm, parents.pop(
                fitness_parents.index(max(fitness_parents)))))
            fitness_parents.pop(fitness_parents.index(max(fitness_parents)))

        fitness_myChildren = [self.getFitness(
            each_sol) for each_sol in myChildren]
        while fixedPop_flag:
            if len(myChildren) == len(parents) or (self.Elitism == 1 and len(myChildren) == len(parents) + 2):
                fixedPop_flag = False
                break
            myChildren.pop(fitness_myChildren.index(min(fitness_myChildren)))
            fitness_myChildren.pop(
                fitness_myChildren.index(min(fitness_myChildren)))

            print(len(parents))

        return myChildren

    def mutate(self, solution):

        newsolution = solution[:]
        if self.probabalistic_decision(self.mutatation_rate) == 1:
            i = 0
            j = 0
            while i == j:
                i = random.randint(0, len(solution) - 1)
                j = random.randint(0, len(solution) - 1)
            temp1 = newsolution[j]
            temp2 = newsolution[i]
            newsolution[i] = temp1
            newsolution[j] = temp2

        return newsolution

    def getFitness(solution):
        score = (len(solution) - 1) * len(solution)
        for i in range(0, len(solution)):
            for j in range(0, len(solution)):
                if i != j:
                    if i - solution[i] == j - solution[j] or i + solution[i] == j + solution[j]:
                        score -= 1
        return score / 2

    def create_crossover(self, solution1, solution2):
        eliminateFlag = True

        # n = int(len(solution1)/2)
        newsolution1 = []
        newsolution2 = []
        if self.crossover_type == 0:
            n = random.randint(0, len(solution1) - 1)
            alleleP1 = solution1[n:]
            alleleP2 = []
            for i in range(len(solution2)):
                eliminateFlag = True
                for j in range(len(alleleP1)):
                    if solution2[i] == alleleP1[j]:
                        eliminateFlag = False
                        break
                if eliminateFlag:
                    alleleP2.append(solution2[i])
            newsolution1 = alleleP2[:n] + alleleP1
            alleleP1 = solution2[n:]
            alleleP2 = []
            for i in range(len(solution1)):
                eliminateFlag = True
                for j in range(len(alleleP1)):
                    if solution1[i] == alleleP1[j]:
                        eliminateFlag = False
                if eliminateFlag:
                    alleleP2.append(solution1[i])
            newsolution2 = alleleP2[:n] + alleleP1
        else:
            n = 0
            k = 0
            while n == k:
                n = random.randint(0, len(solution1) - 1)
                k = random.randint(0, len(solution1) - 1)
            if n > k:
                temp = k
                k = n
                n = temp
            alleleP1 = solution1[n:k + 1]
            alleleP2 = []
            for i in range(len(solution2)):
                eliminateFlag = True
                for j in range(len(alleleP1)):
                    if solution2[i] == alleleP1[j]:
                        eliminateFlag = False
                        break
                if eliminateFlag:
                    alleleP2.append(solution2[i])
            y = 0
            z = 0
            tempchild = []
            for x in range(len(solution1)):
                if x >= n and x <= k:
                    tempchild.append(alleleP1[y])
                    y += 1
                else:
                    tempchild.append(alleleP2[z])
                    z += 1
            newsolution1 = tempchild
            alleleP1 = solution2[n:k + 1]
            alleleP2 = []
            for i in range(len(solution1)):
                eliminateFlag = True
                for j in range(len(alleleP1)):
                    if solution1[i] == alleleP1[j]:
                        eliminateFlag = False
                if eliminateFlag:
                    alleleP2.append(solution1[i])
            y = 0
            z = 0
            tempchild = []
            for x in range(len(solution1)):
                if x >= n and x <= k:
                    tempchild.append(alleleP1[y])
                    y += 1
                else:
                    tempchild.append(alleleP2[z])
                    z += 1
            newsolution2 = tempchild
        return (newsolution1, newsolution2)

    def probabalistic_decision(probability):
        return random.random() < probability

    def create_offspring(self, parents):  # c1 and c2 are chromosomes
        children = []
        children.extend(self.create_crossover(
            GeneticAlgorithm, parents[0], parents[1]))
        children[0] = self.mutate(GeneticAlgorithm, children[0])
        children[1] = self.mutate(GeneticAlgorithm, children[1])

        return(children)

    @ staticmethod
    def runmyGA(self, n, crossoverRate, mutation, elitismSelect, crossoverType, population, generations):
        # Variable Definition we get from the GUI
        self.crossover_rate = crossoverRate
        self.mutatation_rate = mutation
        self.Elitism = elitismSelect
        self.crossover_type = crossoverType

        solutions = []
        # initial population number
        num_solutions = population

        n_queens = n
        generation_limit = generations
        # A random population of candidate solutionsis created
        solutions = self.create_population(num_solutions, n_queens)
        # print("solutions")
        # print(len(solutions))
        sol_fitness = [self.getFitness(each_solution)
                       for each_solution in solutions]
        j = 0

        break_flag = False
        while j < generation_limit:
            for i in range(len(sol_fitness)):
                if sol_fitness[i] == n_queens * (n_queens - 1) / 2:
                    print(solutions[i])
                    print("Fitness: ")
                    print(self.getFitness(solutions[i]))
                    break_flag = True
                    return solutions[i]
                    break
            else:
                print('Generation {}: Max Fitness {}: Sum Fitness {}: Solutions {}:'.format(
                    j, max(sol_fitness), sum(sol_fitness), len(solutions)))
                j += 1
            if break_flag:
                break
            solutions = self.create_children(GeneticAlgorithm, solutions)
            sol_fitness = [self.getFitness(each_solution)
                           for each_solution in solutions]


class BacktrackingSearch:

    global ARCToggle, FCToggle, MCVToggle, MRVToggle, LCVToggle

    global BOAred
    BOAred = []
    ARCToggle = 1
    FCToggle = 1
    LCVToggle = 1
    MCVToggle = 1
    MRVToggle = 1

    def create_Board(n):
        # N is the number of queens
        Board = []
        Board_Rows = []
        for i in range(n):
            for j in range(n):
                Board_Rows.append(0)
            Board.append(Board_Rows)
            Board_Rows = []

        return Board

    def getMRV(board, row):
        # check how many remaining legal values for this variable for (Minimum Remaining Values)
        # if 'Q' not in solution:
        remainingValues = 0
        if(board[row].count('Q') == 0):
            remainingValues = board[row].count(0)

        return remainingValues

    def getMCV(self, board, row):
        # check how many other unassigned variables are constrained by this
        # variable for (Most Constraing Variable)
        ValueConstraints = []
        for i in range(len(board)):
            if self.isSafe(board, row, i):
                x, y = self.getLCV(board, row, i)
                ValueConstraints.append(x)
        return sum(ValueConstraints)

    def getLCV(board, row, col):
        # get the value that rules out the least values from neighboring variables for (Least Constraining Value)
        count = 0
        tempboard = []
        tempboard = deepcopy(board)
        tempboard[row][col] = 'Q'
        print("Queen: ", row + 1)
        # eliminate values in this column on lower side
        for i in range(row, len(board), 1):
            # print("inside upper column side")
            # print("board inside LCV position content: ", tempboard[i][col])
            if tempboard[i][col] == 0 and tempboard[i].count('Q') == 0:
                count += 1
                print("inside lower column side true for column:", col + 1)

        # eliminate values in this column on upper side
        for i in range(row):
            # print("inside upper column side")
            # print("board inside LCV position content: ", tempboard[i][col])
            if tempboard[i][col] == 0 and tempboard[i].count('Q') == 0:
                count += 1
                print("inside upper column side true for column:", col + 1)

        # eliminate values in  lower diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if tempboard[i][j] == 0 and tempboard[i].count('Q') == 0:
                count += 1
                print("inside lower left diagonal")

        # eliminate values in  upper diagonal on right side
        for i, j in zip(range(row, -1, -1), range(col, len(board), 1)):
            if tempboard[i][j] == 0 and tempboard[i].count('Q') == 0:
                count += 1
                print("inside upper right diagonal")

        # eliminate values in  lower diagonal on left side
        for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
            if tempboard[i][j] == 0 and tempboard[i].count('Q') == 0:
                count += 1
                print("inside lower left diagonal")
        # eliminate values in  lower diagonal on right side
        for i, j in zip(range(row, len(board), 1), range(col, len(board), 1)):
            if tempboard[i][j] == 0 and tempboard[i].count('Q') == 0:
                count += 1
                print("inside lower right diagonal")

        return (count, col)

    def printSolution(board):
        for i in range(len(board)):
            for j in range(len(board)):
                print(board[i][j], end=" ")
            print()

    # A utility function to check if a queen can
    # be placed on board[row][col]. Note that this
    # function is called when "col" queens are
    # already placed in columns from 0 to col -1.
    # So we need to check only left side for
    # attacking queens

    def isSafe(board, row, col):

        # Check this element
        if board[row][col] == 1:
            return False
        # check this row
        for i in range(len(board)):
            if board[i][col] == 'Q':
                return False

        # Check this column
        for i in range(len(board)):
            if board[i][col] == 'Q':
                return False

        # Check upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        # Check upper diagonal on right side
        for i, j in zip(range(row, -1, -1), range(col, len(board), 1)):
            if board[i][j] == 'Q':
                return False

        # Check lower diagonal on left side
        for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        # Check lower diagonal on right side
        for i, j in zip(range(row, len(board), 1), range(col, len(board), 1)):
            if board[i][j] == 'Q':
                return False

        return True

    def doFC(board, col, row):
        # eliminate values in this column on lower side
        for i in range(row, len(board), 1):
            if board[i][col] != 'Q':
                board[i][col] = 1

        # eliminate values in this column on upper side
        for i in range(row):
            if board[i][col] != 'Q':
                board[i][col] = 1

        # eliminate values in this row
        for i in range(len(board)):
            if board[row][i] != 'Q':
                board[row][i] = 1

        # eliminate values in  upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] != 'Q':
                board[i][j] = 1

        # eliminate values in  upper diagonal on right side
        for i, j in zip(range(row, -1, -1), range(col, len(board), 1)):
            if board[i][j] != 'Q':
                board[i][j] = 1

        # eliminate values in  lower diagonal on left side
        for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
            if board[i][j] != 'Q':
                board[i][j] = 1

        # eliminate values in  lower diagonal on right side
        for i, j in zip(range(row, len(board), 1), range(col, len(board), 1)):
            if board[i][j] != 'Q':
                board[i][j] = 1

    def doARCCon(self, board, col, row):
        for i in range(len(board)):
            if(board[i].count('Q') == 0):
                if(board[i].count(0) == 0):
                    print("inside arc Consistency queen",
                          i + 1, " has no values")
                    return False
        counter = 0
        for i in range(len(board)):
            if(board[i].count('Q') == 1):
                counter + 1
        if(counter == len(board) - 1):
            print("board after finishing arc Consistency")
            self.printSolution(board)
            return True
        boardtemp = deepcopy(board)
        boardtemp[row][col] = 'Q'

        # eliminate values in this column on lower side and check if the variable has only one value
        for i in range(row, len(boardtemp), 1):
            if boardtemp[i][col] == 0:
                boardtemp[i][col] = 1
                if(boardtemp[i].count(0) == 1):
                    self.doARCCon(self, boardtemp, boardtemp[i].index(0), i)
        #print("Board after eliminating in Consistency lower Column: ")
        # printSolution(boardtemp)

        # eliminate values in this column on upper side and check if the variable has only one value
        for i in range(row):
            if boardtemp[i][col] == 0:
                boardtemp[i][col] = 1
                if(boardtemp[i].count(0) == 1):
                    self.doARCCon(self, boardtemp, boardtemp[i].index(0), i)
        #print("Board after eliminating in Consistency upper Column: ")
        # printSolution(boardtemp)

        # eliminate values in this row
        for i in range(len(boardtemp)):
            if boardtemp[row][i] == 0:
                boardtemp[row][i] = 1

        #print("Board after eliminating in Consistency row: ")
        # printSolution(boardtemp)

        # eliminate values in  upper diagonal on left side and check if the variable has only one value
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if boardtemp[i][j] == 0:
                boardtemp[i][j] = 1
                if(boardtemp[i].count(0) == 1):
                    self.doARCCon(self, boardtemp, boardtemp[i].index(0), i)
        #print("Board after eliminating in Consistency upper left diagonal: ")
        # printSolution(boardtemp)

        # eliminate values in  upper diagonal on right side and check if the variable has only one value
        for i, j in zip(range(row, -1, -1), range(col, len(boardtemp), 1)):
            if boardtemp[i][j] == 0:
                boardtemp[i][j] = 1
                if(boardtemp[i].count(0) == 1):
                    self.doARCCon(self, boardtemp, boardtemp[i].index(0), i)
        #print("Board after eliminating in Consistency upper right diagonal: ")
        # printSolution(boardtemp)

        # eliminate values in  lower diagonal on left side and check if the variable has only one value
        for i, j in zip(range(row, len(boardtemp), 1), range(col, -1, -1)):
            if boardtemp[i][j] == 0:
                boardtemp[i][j] = 1
                if(boardtemp[i].count(0) == 1):
                    self.doARCCon(self, boardtemp, boardtemp[i].index(0), i)
        #print("Board after eliminating in Consistency lower left diagonal: ")
        # printSolution(boardtemp)

        # eliminate values in  lower diagonal on right side
        for i, j in zip(range(row, len(boardtemp), 1), range(col, len(boardtemp), 1)):
            if boardtemp[i][j] == 0:
                boardtemp[i][j] = 1
                if(boardtemp[i].count(0) == 1):
                    self.doARCCon(self, boardtemp, boardtemp[i].index(0), i)
        #print("Board after eliminating in Consistency lower right diagonal: ")
        # printSolution(boardtemp)

        # check if there are any variables with no values
        for i in range(len(boardtemp)):
            if(boardtemp[row].count('Q') == 0):
                if(boardtemp[row].count(0) == 0):
                    return False

        #print("temp Board right before returning true")
        # printSolution(boardtemp)
        board = deepcopy(boardtemp)
        #print("Board right before returning true")
        # printSolution(board)
        return True

    def solveNQUtil(self, board, row):
        global BOAred, ARCToggle, LCVToggle, MCVToggle, FCToggle, MRVToggle
        boardtemp = []
        rowtemp = row
        # base case: If all queens are placed
        # then return true
        if row >= len(board):
            for i in range(len(board)):
                if(board[i].count('Q') == 0):
                    print("solution not found... queen", i + 1,
                          "is not placed we are at queen", row + 1)
                    return False
            print("all queens are placed and the board is set:")
            BOAred = deepcopy(board)
            self.printSolution(board)
            return True
        # saving original board to backtrack to
        boardtemp = deepcopy(board)
        # to measure the MRV,MCV, and LCV respectively
        MRV = []
        MRVIndex = []
        MCV = []
        MCVIndex = []
        constraints = []
        indecies = []

        if (boardtemp[row].count(0) == 0):
            print("inside queen", row + 1, "has no values")
            return False

        for i in range(len(board)):

            # choosing value for the queen we got in recursion
            column = i
            if (LCVToggle == 1):
                print("inside LCV queen: ", row + 1)
                for i in range(len(board)):
                    if self.isSafe(boardtemp, row, i):
                        x, y = self.getLCV(board, row, i)

                        constraints.append(x)
                        indecies.append(y)

                        print("constraints: ")
                        print(constraints)
                        print("index: ")
                        print(indecies)
                        print("index of LCV of queen: ", row + 1)
                        print(indecies[constraints.index(max(constraints))])
                if(len(constraints) != 0):
                    column = indecies[constraints.index(max(constraints))]
                indecies.clear()
                constraints.clear()
            flag = False
            if self.isSafe(boardtemp, row, column):

                # Place this queen in board[i][col]
                boardtemp[row][column] = 'Q'
                if(FCToggle == 1):
                    print("inside Forward Checking")
                    self.doFC(boardtemp, column, row)
                    flag = True
                    for i in range(len(boardtemp)):
                        if(boardtemp[i].count('Q') == 0):
                            if(boardtemp[i].count(0) == 0):
                                flag = False

                if(ARCToggle == 1):
                    print("inside Arc Consistency")
                    print("can we pick ", column + 1,
                          " as position for queen", row + 1, " ?")
                    if(self.doARCCon(self, boardtemp, column, row)):
                        flag = True

                        self.doFC(boardtemp, column, row)
                    print("can we pick ", column + 1,
                          " as position for queen", row + 1, " :", flag)
                if flag is False and (ARCToggle == 1 or FCToggle == 1):

                    boardtemp[row][column] = 1
                    print("board temp after not picking position ",
                          column + 1, " for queen ", row + 1)
                    self.printSolution(boardtemp)
                    continue
                if(FCToggle == 0 and ARCToggle == 0):
                    self.doFC(boardtemp, column, row)
                print("board after placing queen ", row + 1)
                self.printSolution(boardtemp)

                # choosing next variable (queen)
                row += 1
                if (MCVToggle == 1 and row < len(boardtemp)):
                    print("inside MCV")
                    for i in range(len(boardtemp)):
                        print("queen", i + 1)
                        if(boardtemp[i].count('Q') == 0):
                            MCV.append(self.getMCV(self, boardtemp, i))
                            MCVIndex.append(i)

                    row = MCVIndex[MCV.index(max(MCV))]
                    MCV.clear()
                    MCVIndex.clear()

                if(MRVToggle == 1 and row < len(boardtemp)):
                    print("inside MRV Board:")
                    self.printSolution(boardtemp)
                    for i in range(len(boardtemp)):
                        if(boardtemp[i].count('Q') == 0):
                            print("queen ", i + 1,
                                  "is being counted for next assignment")
                            MRV.append(self.getMRV(boardtemp, i))
                            MRVIndex.append(i)
                    print("queen with the minimum remaining values: ")
                    print(MRVIndex[MRV.index(min(MRV))] + 1)
                    print("Remaining Values:")
                    print(MRV)
                    print("Queens:")
                    print(MRVIndex)
                    row = MRVIndex[MRV.index(min(MRV))]
                    MRV.clear()
                    MRVIndex.clear()

                # recur to place rest of the queens
                if self.solveNQUtil(self, boardtemp, row) is True:
                    return True

                # position leads to no solution
                if((MRVToggle == 1 or MCVToggle == 1)):
                    row = rowtemp
                if((MRVToggle != 1 and MCVToggle != 1)):
                    row -= 1

            boardtemp = deepcopy(board)
            boardtemp[row][column] = 1
            board = deepcopy(boardtemp)

            print("board temp for queen ", row + 1, " position: ", column + 1)
            self.printSolution(boardtemp)
            if (boardtemp[row].count(0) == 0):
                print("inside count == 0 for queen ", row + 1, "after isSafe:")
                self.printSolution(boardtemp)
                return False
            # removing position from LCV to try another value

        # if the queen can not be placed in any row in
        # this colum col then return false
        return False

    # This function solves the N Queen problem using
    # Backtracking. It mainly uses solveNQUtil() to
    # solve the problem. It returns false if queens
    # cannot be placed, otherwise return true and
    # placement of queens in the form of 1s.
    # note that there may be more than one
    # solutions, this function prints one of the
    # feasible solutions.

    def solveNQ(self, N):
        board = self.create_Board(N)
        global ARCToggle, FCToggle, MCVToggle, MRVToggle, LCVToggle
        if self.solveNQUtil(self, board, 0) == False:
            print("Solution does not exist")
            return None
        print("----------------------------------------------------------")
        print("FINAL SOLUTION")
        self.printSolution(BOAred)
        print("----------------------------------------------------------")
        sol = []
        for i in range(len(BOAred)):
            sol.append(BOAred[i].index('Q'))
        return sol

    @ staticmethod
    def runmyBTA(self, n, ARC_Toggle, FC_Toggle, LCV_Toggle, MCV_Toggle, MRV_Toggle):
        # Variable Definition we get from the GUI
        self.ARCToggle = ARC_Toggle
        self.FCToggle = FC_Toggle
        self.LCVToggle = LCV_Toggle
        self.MCVToggle = MCV_Toggle
        self.MRVToggle = MRV_Toggle
        print("ARCToggle: ", ARC_Toggle)
        print("FCToggle: ", FC_Toggle)
        print("LCVToggle: ", LCV_Toggle)
        print("MCVToggle: ", MCV_Toggle)
        print("MRVToggle: ", MRV_Toggle)
        # create board
        n_queens = n
        print("queens inside runmyBTA: " + str(n_queens))
        solution = self.create_Board(n_queens)
        self.printSolution(solution)
        # begin backtracking starting from the first queen
        sol = []
        sol = deepcopy(self.solveNQ(self, n_queens))
        if sol is not None:
            return sol
        return None
        # if MCV is true do getRemainingConstraints(board) to pick the next variable to assign
        # if MRV is true do  getRemainingValues(board) to pick the next variable to assign
        # if LCV is true do getLCV(board) to pick the value for the current variable
        # if FCToggle is true trigger doFC(board) to eliminate inconsistent values in neighboring queens (that can be affected by this assignment)
        # if ARCToggle is true trigger doARC(board) to eliminate inconsistent values in neighboring queens (that can be affected by this assignment. it is recursive until all variables have a consistent arc)


class NQueens:
    """
    This class represents the N-Queens problem.
    There is no UI, but its methods and attributes can be used by a GUI.
    """

    def __init__(self, n):
        self._size = n
        self.reset_board()

    def get_size(self):
        """
        Get size of board (square so only one value)
        """
        return self._size

    def reset_new_size(self, value):
        """
        Resets the board with new dimensions (square so only one value).
        """
        self._size = value
        self.reset_board()

    def get_board(self):
        """
        Get game board.
        """
        return self._board

    def reset_board(self):
        """
        Restores board to empty, with current dimensions.
        """
        self._board = [[EMPTY_SPOT] * self._size for _ in range(self._size)]

    def is_winning_position(self):
        """
        Checks whether all queens are placed by counting them. There should be as many as the board size.
        """
        num_queens = sum(row.count(QUEEN) for row in self._board)
        return num_queens >= self._size

    def place_queen(self, pos):
        """
        Add a queen (represented by 1) at a given (row, col).
        """
        if self.is_legal_move(pos):
            self._board[pos[0]][pos[1]] = QUEEN
            return True  # Return value is useful for GUI - e.g trigger sound.
        return False

    def is_legal_move(self, pos):
        """
        Check if position is on board and there are no clashes with existing queens
        """
        return self.check_row(pos[EMPTY_SPOT]) and self.check_cols(pos[1]) and self.check_diagonals(pos)

    def check_row(self, row_num):
        """
        Check a given row for collisions. Returns True if move is legal
        """
        return not QUEEN in self._board[row_num]

    def check_cols(self, pos):
        """
        Check columns and return True if move is legal, False otherwise
        """
        legal = True
        for row in self._board:
            if row[pos] == QUEEN:
                legal = False
        return legal

    def check_diagonals(self, pos):
        """
        Checks all 4 diagonals from given position in a 2d list separately, to determine
        if there is a collision with another queen.
        Returns True if move is legal, else False.
        """
        num_rows, num_cols = len(self._board), len(self._board[0])
        row_num, col_num = pos

        # Lower-right diagonal from (row_num, col_num)
        # This covers case where spot is already occupied.
        i, j = row_num, col_num
        while i < num_rows and j < num_cols:
            if self._board[i][j] == QUEEN:
                return False
            i, j = i + 1, j + 1

        # Upper-left diagonal from (row_num, col_num)
        i, j = row_num - 1, col_num - 1
        while i >= 0 and j >= 0:
            if self._board[i][j] == QUEEN:
                return False
            i, j = i - 1, j - 1

        # Upper-right diagonal from (row_num, col_num)
        i, j = row_num - 1, col_num + 1
        while i >= 0 and j < num_cols:
            if self._board[i][j] == QUEEN:
                return False
            i, j = i - 1, j + 1

        # Lower-left diagonal from (row_num, col_num)
        i, j = row_num + 1, col_num - 1
        while i < num_cols and j >= 0:
            if self._board[i][j] == QUEEN:
                return False
            i, j = i + 1, j - 1

        return True


class NQueensGUI:
    """
    GUI for N-Queens game.
    """

    INPUT_1 = ''
    INPUT_2 = ''
    queen_image = simplegui.load_image(
        "https://compucademy.net/assets/queen.PNG")
    queen_image_size = (queen_image.get_width(), queen_image.get_height())
    FRAME_SIZE = (1000, 1000)
    BOARD_SIZE = 20  # Rows/cols

    def __init__(self, game):
        """
        Instantiate the GUI for N-Queens game.
        """
        # Game board
        self._game = game
        self._size = game.get_size()
        self._square_size = self.FRAME_SIZE[0] // self._size
        # genetic algorithm options
        self.crossoverToggle = False
        self.elitismToggle = False
        self.population_size = 16
        self.generation_limit = 100
        self.mutation_rate = 0.05
        self.crossover_rate = 1
        # backtracking toggle options
        self.ARCToggle = 1
        self.FCToggle = 1
        self.LCVToggle = 1
        self.MCVToggle = 1
        self.MRVToggle = 1

        # self.

        # Set up frame
        self.setup_frame()

    def setup_frame(self):
        """
        Create GUI frame and add handlers.
        """
        self._frame = simplegui.create_frame("N-Queens Game",
                                             self.FRAME_SIZE[0], self.FRAME_SIZE[1])
        self._frame.set_canvas_background('white')

        # Set handlers
        self._frame.set_draw_handler(self.draw)
        self._frame.add_label("Welcome to N-Queens")
        self._frame.add_label("")  # For better spacing.
        # Board Size
        board_msg = "Current board size: " + str(self._size)
        # For better spacing.
        self._size_label = self._frame.add_label(board_msg)
        self._frame.add_input('', self.BOARD_SIZE_handler, 50)

        # Genetic Search:
        self._frame.add_label("")
        self._frame.add_label("Genetic Algorithm:")
        # population
        population_msg = "inital Population(>= 2): " + \
            str(self.population_size) + " (Currently)"
        self._population_label = self._frame.add_label(population_msg)
        self._frame.add_input('', self.population_handler, 50)
        self._frame.add_label("")  # For better spacing.
        # generation_limit
        generation_limit_msg = "Generation limit (>= 2): " + \
            str(self.generation_limit) + " (Currently)"
        self._generation_limit_label = self._frame.add_label(
            generation_limit_msg)
        self._frame.add_input('', self.generations_handler, 50)
        self._frame.add_label("")  # For better spacing.
        # mutation rate
        mutation_rate_msg = "mutation rate (0.000 - 1.000): " + \
            str(self.mutation_rate) + " (Currently)"
        self._mutation_rate_label = self._frame.add_label(mutation_rate_msg)
        self._frame.add_input('', self.mutation_handler, 50)
        self._frame.add_label("")  # For better spacing.
        # crossover rate
        crossover_rate_msg = "crossover rate (0.000 - 1.000): " + \
            str(self.crossover_rate) + " (Currently)"
        self._crossover_rate_label = self._frame.add_label(crossover_rate_msg)
        self._frame.add_input('', self.crossover_handler, 50)
        self._frame.add_label("")  # For better spacing.
        # crossover Toggle
        self._crossoverToggle_input = self._frame.add_button(
            "crossover: single", self.toggleCrossover)
        self._ElitismToggle_input = self._frame.add_button(
            "Elitism: OFF", self.toggleElitism)
        self._frame.add_button("run genetic search", self.runGeneticSearch)
        self._frame.add_label("")

        # A* search:
        self._frame.add_label("A* search:")
        self._frame.add_button("run A* search", self.runAstarSearch)
        self._frame.add_label("")

        # Backtracking Search:
        self._frame.add_label("Backtracking Search:")
        self._ARCToggle_input = self._frame.add_button(
            "Arc Consistency: ON", self.toggleARC)
        self._FCToggle_input = self._frame.add_button(
            "Forward Checking: ON", self.toggleFC)
        self._frame.add_label("Value Ordering:")
        self._LCVToggle_input = self._frame.add_button(
            "Least Constraining Value: ON", self.toggleLCV)
        self._frame.add_label("Variable Ordering")
        self._MCVToggle_input = self._frame.add_button(
            "Degree Heuristic: ON", self.toggleMCV)
        self._frame.add_label("")
        self._MRVToggle_input = self._frame.add_button(
            "Minimum Remaining Values: ON", self.toggleMRV)
        self._frame.add_button("run Backtrack search", self.runBacktrackSearch)
        self._frame.add_label("")
        # restart button
        self._frame.add_button("Reset", self.reset)
        self._frame.add_label("")  # For better spacing.

        self._frame.add_label("")  # For better spacing.

        self._frame.add_label("")  # For better spacing.
        self._label = self._frame.add_label("")
        self._frame.add_label("")
        self._label1 = self._frame.add_label("")
        self._label2 = self._frame.add_label("")
        self._label3 = self._frame.add_label("")
        self._label4 = self._frame.add_label("")
        self._label5 = self._frame.add_label("")
        self._label6 = self._frame.add_label("")
        self._label7 = self._frame.add_label("")
        self._label8 = self._frame.add_label("")
        self._label9 = self._frame.add_label("")
        self._label10 = self._frame.add_label("")

    def BOARD_SIZE_handler(self, text):  # type: (str) -> None
        """gets n of queens from textfield"""
        new_size = int(text)
        self._game.reset_new_size(new_size)
        self._size = self._game.get_size()
        self._square_size = self.FRAME_SIZE[0] // self._size
        board_msg = "board size: " + str(self._size) + " (Currently)"
        self._size_label.set_text(board_msg)
        self.reset()
        # print(self._game.get_size())

    # Genetic search options handler
    def population_handler(self, text):  # type: (str) -> None
        """gets initial population size from textfield"""
        self.population_size = int(text)
        self.population_msg = "inital Population(>= 2): " + \
            str(self.population_size) + " (Currently)"
        self._population_label.set_text(self.population_msg)
        print(self.population_size)

    def generations_handler(self, text):  # type: (str) -> None
        """gets generation_limit from textfield"""

        self.generation_limit = int(text)
        self.generation_limit_msg = "Generation limit (>= 2): " + \
            str(self.generation_limit) + " (Currently)"
        self._generation_limit_label.set_text(self.generation_limit_msg)
        print(self.generation_limit)

    def mutation_handler(self, text):  # type: (str) -> None
        """gets mutation rate from textfield"""
        self.mutation_rate = float(text)
        self.mutation_rate_msg = "mutation rate (0.000 - 1.000): " + \
            str(self.mutation_rate) + " (Currently)"
        self._mutation_rate_label.set_text(self.mutation_rate_msg)
        print(self.mutation_rate)

    def crossover_handler(self, text):  # type: (str) -> None
        """gets crossover rate from textfield"""
        # pylint: disable=global-statement
        self.crossover_rate = float(text)
        self.crossover_rate_msg = "crossover rate (0.000 - 1.000): " + \
            str(self.crossover_rate) + " (Currently)"
        self._crossover_rate_label.set_text(self.crossover_rate_msg)
        print(self.crossover_rate)

    def toggleCrossover(self):
        """ switching between single point and multi point crossover"""
        if self.crossoverToggle:
            self.crossoverToggle = False
            self._crossoverToggle_input.set_text("crossover: Single")
        else:
            self.crossoverToggle = True
            self._crossoverToggle_input.set_text("crossover: Multi")

    def toggleElitism(self):
        """ switching Elitism ON and OFF"""
        if self.elitismToggle:
            self.elitismToggle = False
            self._ElitismToggle_input.set_text("Elitism: OFF")
        else:
            self.elitismToggle = True
            self._ElitismToggle_input.set_text("Elitism: ON")

    # Backtracking options handler

    def toggleARC(self):
        """ switching between single point and multi point crossover"""
        if self.ARCToggle:
            self.ARCToggle = 0
            self._ARCToggle_input.set_text("Arc Consistency: OFF")
        else:
            self.ARCToggle = 1
            self._ARCToggle_input.set_text("Arc Consistency: ON")

    def toggleFC(self):
        """ switching Elitism ON and OFF"""
        if self.FCToggle:
            self.FCToggle = 0
            self._FCToggle_input.set_text("Forward Checking: OFF")
        else:
            self.FCToggle = 1
            self._FCToggle_input.set_text("Forward Checking: ON")

    def toggleLCV(self):
        """ switching between single point and multi point crossover"""
        if self.LCVToggle:
            self.LCVToggle = 0
            self._LCVToggle_input.set_text("Least Constraining value: OFF")
        else:
            self.LCVToggle = 1
            self._LCVToggle_input.set_text("Least Constraining value: ON")

    def toggleMRV(self):
        """ switching Elitism ON and OFF"""
        if self.MRVToggle:
            self.MRVToggle = 0
            self._MRVToggle_input.set_text("Minimum Remaining Values: OFF")
        else:
            self.MRVToggle = 1
            self._MRVToggle_input.set_text("Minimum Remaining Values: ON")

    def toggleMCV(self):
        """ switching Elitism ON and OFF"""
        if self.MCVToggle:
            self.MCVToggle = 0
            self._MCVToggle_input.set_text("Degree Heuristic: OFF")
        else:
            self.MCVToggle = 1
            self._MCVToggle_input.set_text("Degree Heuristic: ON")
    # GUI Methods

    def start(self):
        """
        Start the GUI.
        """
        self._frame.start()

    def reset(self):
        """
        Reset the board
        """
        self._game.reset_board()
        self._label.set_text("")
        self._label1.set_text("")
        self._label2.set_text("")
        self._label3.set_text("")
        self._label4.set_text("")
        self._label5.set_text("")
        self._label6.set_text("")
        self._label7.set_text("")
        self._label8.set_text("")
        self._label9.set_text("")
        self._label10.set_text("")

    def runGeneticSearch(self):
        sol = []
        self.reset()
        tic = time.perf_counter()
        sol = GeneticAlgorithm.runmyGA(GeneticAlgorithm, self._size, self.crossover_rate, self.mutation_rate,
                                       self.elitismToggle, self.crossoverToggle, self.population_size, self.generation_limit)
        toc = time.perf_counter()

        if sol is not None:
            for i in range(len(sol)):
                # placing the queens on the board from the optimal solution
                self._game.place_queen((i, sol[i]))

            if self._game.is_winning_position():
                success_sound.play()

                timemsg = "Time: " + str(toc - tic) + " seconds"
                self._label.set_text("Well done. You have found a solution.")
                self._label1.set_text("Algorithm: Genetic Algorithm")
                pop = "Popultaion Size: " + str(self.population_size)
                self._label2.set_text(pop)
                gens = "Generation Limit: " + str(self.generation_limit)
                self._label3.set_text(gens)
                mutate = "mutation rate: " + str(self.mutation_rate)
                self._label4.set_text(mutate)
                cross = "crossover rate: " + str(self.crossover_rate)
                self._label5.set_text(cross)
                self._label6.set_text(timemsg)
        else:
            noSolution = "No solution found before the generation limit: " + \
                str(self.generation_limit)
            self._label.set_text(noSolution)

    def runAstarSearch(self):
        self.reset()
        sol = []
        tic = time.perf_counter()
        solAndNodes = AstarSearch.run_AstarSearch(AstarSearch, self._size)
        sol = solAndNodes[0]
        NodesExpanded = solAndNodes[1]
        toc = time.perf_counter()

        for i in range(len(sol)):

            self._game.place_queen((i, sol[i]))
        if self._game.is_winning_position():
            success_sound.play()
            self._label.set_text("Well done. You have found a solution.")
            self._label1.set_text("Algorithm: A* Search")
            nodesMsg = "Nodes Expanded: " + str(NodesExpanded)
            self._label2.set_text(nodesMsg)
            timemsg = "Time: " + str(toc - tic) + " seconds"
            self._label3.set_text(timemsg)

    def runBacktrackSearch(self):
        sol = []
        self.reset()
        tic = time.perf_counter()
        print("queens: ")
        print(self._size)
        sol = BacktrackingSearch.runmyBTA(
            BacktrackingSearch, self._size, self.ARCToggle, self.FCToggle, self.LCVToggle, self.MCVToggle, self.MRVToggle)
        toc = time.perf_counter()

        if sol is not None:
            for i in range(len(sol)):
                # placing the queens on the board from the optimal solution
                self._game.place_queen((i, sol[i]))

            if self._game.is_winning_position():
                success_sound.play()
                timemsg = "Time: " + str(toc - tic) + " seconds"
                FCmsg = "Forward Checking: " + str(self.FCToggle)
                ACmsg = "Arc Consistency: " + str(self.FCToggle)
                LCVmsg = "Least Constraining Value (LCV): " + \
                    str(self.LCVToggle)
                MCVmsg = "Most Constraining Variable (MCV): " + \
                    str(self.MCVToggle)
                MRVmsg = "Minimum Remaining Values (MRV): " + \
                    str(self.LCVToggle)
                self._label.set_text("Well done. You have found a solution.")
                self._label1.set_text("Algorithm: Backtracking Search ")
                self._label2.set_text("Filtering: ")
                self._label3.set_text(FCmsg)
                self._label4.set_text(ACmsg)
                self._label5.set_text("Value ordering:")
                self._label6.set_text(LCVmsg)
                self._label7.set_text("Variable ordering:")
                self._label8.set_text(MCVmsg)
                self._label9.set_text(MRVmsg)
                self._label10.set_text(timemsg)
        else:
            noSolution = "No solution found before the generation limit: " + \
                str(self.generation_limit)
            self._label.set_text(noSolution)

    def draw(self, canvas):
        """
        Draw handler for GUI.
        """
        board = self._game.get_board()
        dimension = self._size
        size = self._square_size

        # Draw the squares
        for i in range(dimension):
            for j in range(dimension):
                color = "green" if ((i % 2 == 0 and j % 2 == 0) or i %
                                    2 == 1 and j % 2 == 1) else "red"
                points = [(j * size, i * size), ((j + 1) * size, i * size), ((j + 1) * size, (i + 1) * size),
                          (j * size, (i + 1) * size)]
                canvas.draw_polygon(points, 1, color, color)

                if board[i][j] == 1:
                    canvas.draw_image(
                        self.queen_image,  # The image source
                        (self.queen_image_size[0] // 2,
                         self.queen_image_size[1] // 2),
                        # Position of the center of the source image
                        self.queen_image_size,  # width and height of source
                        ((j * size) + size // 2, (i * size) + size // 2),
                        # Where the center of the image should be drawn on the canvas
                        (size, size)  # Size of how the image should be drawn
                    )


gui = NQueensGUI(NQueens(BOARD_SIZE))
gui.start()
