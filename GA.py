# Genetic Algorithm (just for fun)

# Take the nominal forms of a list data type and within a given vector length gain the algorithm result
# through selection, crossover, mutation.

# Lets take this demostration as a way to formulate it by the knapsack problem.

import numpy as np
import math
import random
from time import process_time_ns

# Generally keep the data set within an approximate range (100, 1000)
# Now depending on the conditions set towards the algorithm we should incorporate any kind of scaling
# with the datas used, we just need to make sure that we include a weight for the best possible measurements

# Let the weight Distribution be a bag full of whatever

#random.seed(10)

# Modify the weight distribution if you think the value of something else is more important than the other.
maximumCarriage = 1000
weightDistribution = []
weightMax = 0.0

# GA Settings
numberOfWeights = 20
populationSize = 200
tournamentSize = 5
spinStopsSize = 4
generationSize = 100
mutationRate = 0.1
crossoverRate = 0.1

#Adjust accordingly
weightDiversionMin = 0.01
weightDiversionMax = 0.3

def generateDataSet(populationSize, vectorLength):
    pop = []
    for n in range(populationSize):
        vector = []
        for m in range(vectorLength):
            vector.append(random.randint(0, 1))
        pop.append(vector)
    return np.array(list(pop));

def generateWeightSet(size):
    global weightDistribution
    global weightMax
    weights = []
    weightDisparity = 1.0
    numberOfTries = 0
    #while(weightTotal(weights) < 1.0 and not(weightTotal(weights) > 0.9)):
    '''
    while(weightTotal(weights) < weightDisparity and not (weightTotal(weights) > weightDisparity-0.1)):
        for i in range(size):
            weights.append(random.uniform(weightDiversionMin, weightDiversionMax))
        if(weightTotal(weights) < weightDisparity):
            break
        else:
            numberOfTries += 1;
            weights.clear()
    '''
    for i in range(size):
        weights.append(random.uniform(weightDiversionMin, weightDiversionMax))
    print(f"Number of Tries before Successful Weight Distribution: {numberOfTries}")
    print(f"Weight Distribution: {weights}")
    print(f"Weight Total: {weightTotal(weights)} of the Max-{maximumCarriage*weightTotal(weights)}")
    weightMax = maximumCarriage*weightTotal(weights)
    weightDistribution = weights

def weightTotal(weights):
    if (len(weights) == 0):
        return 0.0
    else:
        addedWeights = 0.0
        for n in weights:
            addedWeights += n
        return addedWeights

class GeneticAlgorithm():                                       # There are many variables being used, maybe memory conservation could be improved
    def __init__(self, population, generations):
        self.originalPopulation = population
        self.population = population
        self.generations = generations
    def start(self):
        for i in range(self.generations):
            fitnesses = self.fitnessEvaluation()
            best = fitnesses[0]
            bestIndex = 0
            for index, n in enumerate(fitnesses): # Stop Condition
                if(n > best):
                    best = n
                    bestIndex = index
            if(best == maximumCarriage):
                print("\nThe Genetic Algorithm has reached an Optimum Solution " 
                      "with the Following weights: \n{}\n".format(weightDistribution))
                self.printCurrentStatus(fitnesses, i, best, bestIndex)
                break
            self.printCurrentStatus(fitnesses, i, best, bestIndex)
            new_pop = []
            for j in range(math.floor(len(self.population)/2)):
                #parent1, parent2 = self.stochasticUniversalSampling(spinStopsSize, fitnesses)
                parent1, parent2 = self.tournamentSelection(tournamentSize, fitnesses)
                child1, child2 = self.uniformCrossover(parent1, parent2)
                new_pop.append(self.bitFlipMutation(child1))
                new_pop.append(self.bitFlipMutation(child2))
            self.population = np.array(list(new_pop))
    def printCurrentStatus(self, fitnesses, currentGeneration, best, bestIndex):
        print(f"Current Generation: {currentGeneration+1}")
        print(f"Best Individual: {self.population[bestIndex]}")
        print(f"Best Fitness: {best}")
        #print(f"Fitnesses: {fitnesses}\n")
        #print(f"Max Fitness Possible: {weightMax}\n")
        print(f"Max Fitness Possible: 1000")
    def fitnessEvaluation(self):
        fitnesses = []
        for n in self.population:
            approxTotal = 0
            for index, m in enumerate(n):
                approxTotal += int(m*maximumCarriage*weightDistribution[index]) # Gene value * Maximum luggage Weight * weight at Gene
            if(approxTotal > maximumCarriage):
                fitnesses.append(0)
            elif(approxTotal == 0):
                fitnesses.append(0)
            else:
                fitnesses.append(approxTotal)
        return np.array(fitnesses)
    def tournamentSelection(self, tournamentSize, fitnesses):
        parents = []
        for p in range(2):
            randomGladiatorNumber = random.randint(0, len(self.population)-1)
            randomGladiator = self.population[randomGladiatorNumber]
            best = self.population[randomGladiatorNumber]
            for i in range(tournamentSize):
                randomOpponentNumber = random.randint(0, len(self.population)-1)
                if(fitnesses[randomOpponentNumber] > fitnesses[randomGladiatorNumber]):
                    best = self.population[randomOpponentNumber]
            parents.append(best)
        return (parents[0], parents[1])
    def stochasticUniversalSampling(self, numberOfPoints, fitnesses): # Not Very Optimized Should look back here for improvements
        parents = []        # Only Two Parents, Wouldn't make sense to use more than that.
        for p in range(2):  # O(n^3) ??? Could be bad probably needs optimization, spinning randomly is honestly very inefficient
            total = 0
            for i in range(len(self.population)):   # Get Total
                total += fitnesses[i]
            randomNumbers = []
            for i in range(numberOfPoints):   # Random numbers serve as spin stops
                randomNumbers.append(random.randint(0, total))
            indexes = []
            for n in randomNumbers:     # Spin to find random Numbers
                addedTotal = 0
                for index, m in enumerate(fitnesses): # Very inefficient but works well
                    addedTotal += m
                    if(index+1 == len(fitnesses)):
                        indexes.append(index)
                        break
                    if(n < fitnesses[0]):
                        indexes.append(0)
                        break
                    if(addedTotal > n):
                        indexes.append(index-1)
                        break
            #print(f"Random Number: {randomNumbers}")
            #print(f"Indexes: {indexes}")
            if(len(indexes) != len(randomNumbers)): # We need to ensure the 2 list have equal shapes
                print(f"Random Number: {randomNumbers}")
                print(f"Indexes: {indexes}")
                raise
            best = indexes[0]
            '''
            for index, i in enumerate(indexes):   # Compare to find the best one
                if(index+1 == len(indexes)):
                    break
                else:
                    if(fitnesses[index+1] > fitnesses[index]):
                        best = indexes[index+1]
            '''
            #parents.append(self.population[best])
            parents.append(self.population[random.choice(indexes)])
        return (parents[0], parents[1])
    def uniformCrossover(self, parent1, parent2):
        child1, child2 = (parent1, parent2)
        for n in child1:
            if(crossoverRate > random.uniform(0.0, 1.0)):
                temp = child1[n]
                child1[n] = child2[n]
                child2[n] = temp
        return (child1, child2)
    def bitFlipMutation(self, ind):
        flipedInd = ind
        for index, n in enumerate(flipedInd):
            if(mutationRate > random.uniform(0.0, 1.0)):
                if(n == 0):
                    flipedInd[index] = 1
                elif(n == 1):
                    flipedInd[index] = 0
                else:
                    raise
        return flipedInd

if __name__ == "__main__":
    print("Randomizing Sample Set for Converging...")
    start = process_time_ns()
    generateWeightSet(numberOfWeights)
    x = generateDataSet(populationSize, len(weightDistribution))
    #print(x.shape())
    print(x)
    end = process_time_ns()
    print(f"Time Accumulation of sample set: {float((end-start)/1000000)} ms\n")
    print("Now Performing Genetic Algorithm Functions")
    start = process_time_ns()
    someGeneticAlgorithm = GeneticAlgorithm(x, generationSize)
    someGeneticAlgorithm.start()
    end = process_time_ns()
    print(f"Time Accumulation of Genetic Algorithm: {float((end-start)/1000000000)} s\n")



