import numpy as np
import math
import streamlit as st

class GA:


    def __init__(self,df,dist_table,cendis_table,info_table,cars,weightlimit,oplimit,rng,bar,n=100,genes=20,rparents=9.0/20.0,rchildren=9.0/40.0,rmutation=1.0/10.0):
        
        # Instance Variable
        self.df = df.copy()
        self.dist_table = dist_table
        self.cendis_table = cendis_table
        self.info_table = info_table
        self.options = info_table.index.values #array
        self.noptions = len(self.options)
        self.rng = rng
        self.bar = bar

        self.cars = cars
        self.n = n
        self.genes = genes
        self.rparents = rparents
        self.rchildren = rchildren
        self.rmutation = rmutation
        self.nparents = round(self.rparents*self.n/2)*2
        self.nchildren = self.nparents//2
        self.nOld = self.n - (self.nparents + self.nchildren)
        self.nmutation = round(rmutation*self.n)

        self.weightlimit = weightlimit
        self.oplimit = oplimit
        self.limitref_w = info_table.loc[:,'sum'].sum()/self.cars
        self.infostd = info_table.loc[:,'sum'].std()
        self.flex = {
            "min" : max(self.weightlimit[0],self.limitref_w - self.infostd),
            "max": min(self.weightlimit[1],self.limitref_w + self.infostd)
        }
        self.mile = np.empty([self.n])

        self.Tour = np.empty([self.n,len(self.options)],dtype=object)
        print("size Tour")
        print(self.Tour.shape)
        self.TourParents = np.empty([self.nparents,len(self.options)],dtype=object)
        self.TourChildren = np.empty([self.nchildren,len(self.options)],dtype=object)
        self.CarCuts = np.empty([self.n,self.cars],dtype=object)
        self.popFitness = None
        self.bestOfBest = {
            "fitness" : math.inf,
            "Tour" : None,
            "Cars": None,
            "epoch": 0
        }

        self.stage = 0


    def fillBest(self,fitness,array,carCuts,epoch):

        self.bestOfBest = {
            "fitness" : fitness[0],
            "Tour" : array[0],
            "Cars": carCuts[0],
            "epoch": epoch + 1
        }


    def distanceFromCendis(self,path):

        distance = 0
        distance = distance + float(self.cendis_table.loc[path[0]])
        for node in range(len(path)-1):
            distance = distance + float(self.dist_table.loc[path[node],path[node+1]])
        return distance

    
    def orderFromCendis(self,path):

        route = self.cendis_table.loc[path].sort_values(by=["CENDIS"]).index.values

        for edge in range(1,len(route)-1):

            reference = route[edge-1]
            route[edge:] = self.dist_table.loc[(route[edge:]),[reference]].sort_values(by=[reference]).index.values

        return route


    def TourDistanceFitness(self,carCuts,Tour):

        distance = 0
        ncars = len(carCuts)
        car = 0

        while (((car+1)<=ncars) and not (carCuts[car] is None)):
            inicio = carCuts[car]["inicio"]
            fin = carCuts[car]["fin"]

            path = Tour[inicio:fin+1]
            distance += self.distanceFromCendis(path)
            car += 1
        return distance


    def carDistanceFitness(self,carCuts,Tour):

        ncars = len(carCuts)

        distance = np.empty([ncars])
        car = 0

        while (((car+1)<=ncars) and not (carCuts[car] is None)):

            inicio = carCuts[car]["inicio"]
            fin = carCuts[car]["fin"]

            path = Tour[inicio:fin+1]

            distance[car] = self.distanceFromCendis(path)
            car += 1


        mean = np.mean(distance)
        std = np.std(distance)

        return mean, std


    def opMissing(self,carCuts,Tour):

        last = -1

        LastCar = False

        while not LastCar:

            if carCuts[last] is not None:
                fin = carCuts[last]["fin"]
                path = Tour[fin+1:]
                LastCar = True
                op, _ = self.weighting(path)

            last -= 1
        return op


    def weighting(self,path):

        op = 0
        weight = 0.0
        for node in range(len(path)):
            op += int(self.info_table.loc[path[node],"count"])
            weight += float(self.info_table.loc[path[node],"sum"])
        return op,weight


    def filtering(self,path,availableWeight,availableOp):

        weightCondition = self.info_table.loc[path,"sum"] <= availableWeight
        opCondition = self.info_table.loc[path,"count"] <= availableOp

        options = np.extract((weightCondition) & (opCondition), path)
        return options


    def NI(self,path,reference):

        shortest = math.inf
        for node in range(len(path)):
           distance = float(self.dist_table.loc[path[node],reference])
           if distance < shortest:
               shortest = distance
               index = node
        return index


    def fillCar(self,inicio,fin,op,peso):

        carInfo = {
                "inicio": inicio,
                "fin": fin,
                "op": op,
                "peso": peso
            }
        return carInfo


    def updateCar(self,sample,carInfo):

        inicio = carInfo["inicio"]
        fin = carInfo["fin"]
        opW, weighW = self.weighting(self.Tour[sample][inicio:fin+1])
        carInfo = self.fillCar(inicio,fin,opW,weighW) 
        return carInfo


    def carDist(self,sample,car,carInfo):

        self.CarCuts[sample][car] = carInfo.copy()


    def removeInQ(self,nextPoint):

        indexToRemove = np.where(self.subQueue == nextPoint)
        self.subQueue = np.delete(self.subQueue, indexToRemove)


    def fitness(self,carCuts,Tour):
        Ncars = len(carCuts)
        TotalDistance = self.TourDistanceFitness(carCuts,Tour)
        Mean, Std = self.carDistanceFitness(carCuts,Tour)
        AvgxStd = Mean * Std
        opMissing = self.opMissing(carCuts,Tour) + 1

        fitness = ((TotalDistance)*opMissing)*((AvgxStd)*Ncars)
        
        return fitness


    def totalFitness(self,carCuts,Tour):

        fitness = np.empty([len(Tour)])

        for sample in range(len(Tour)):
            fitness[sample] = self.fitness(carCuts[sample],Tour[sample])

        return fitness


    def updatePopFitness(self):
        
        self.popFitness = self.totalFitness(self.CarCuts,self.Tour)


    def fitnessMean(self):
        return self.popFitness.mean()


    def bestFitness(self,n):
        
        sortedFitness = np.argsort(self.popFitness)
        return sortedFitness[:n]
         

    def crossover(self,parents):

        chromosomes = np.copy(parents)
        lenChromosomes = chromosomes.shape[1]
        crossoverPoint = lenChromosomes//2

        firstSection = chromosomes[0][:crossoverPoint]
        secondSection = np.setdiff1d(chromosomes[1],firstSection, assume_unique=True)
        Child = np.copy(firstSection)

        for sample in range(len(secondSection)):

            reference = secondSection[sample]
            idPosition = self.NI(Child,reference)

            if self.rng.uniform(0,1) >0.5:
                idPosition += 1
            Child = np.insert(Child,idPosition,reference)

        return Child


    def diff2d(self,A,B):

        a = set(tuple(arr) for arr in A)
        b = set(tuple(arr) for arr in B)
        
        c = a - b

        return np.array([np.asarray(arr) for arr in c],dtype=object)

    
    def oldPopulation(self):
        
        oldNoParents = self.diff2d(self.Tour,self.TourParents)
        self.rng.shuffle(oldNoParents)
        return oldNoParents[:self.nOld]

    
    def mutate(self,TourM,optionslen,rate=0.2):

        Tour = np.copy(TourM)
        vectorlen = round(optionslen * rate)
        lastMutationPoint = optionslen - vectorlen
        index = self.rng.integers(0,lastMutationPoint)
        Tour[index:index+vectorlen] = np.flip(Tour[index:index+vectorlen])

        return Tour


    def cutter(self,sample,Tour):

        car = 0
        opW, weightW = self.weighting(Tour[:1])
        carInfo = self.fillCar(0,0,opW,weightW)
        
        for index in range(1,len(Tour)):

            opW, weightW = self.weighting(Tour[carInfo["inicio"]:index+1])

            if  opW <= self.oplimit and weightW <= self.flex["max"]:

                carInfo["fin"] = index
                carInfo = self.updateCar(sample,carInfo)

            else:

                self.carDist(sample,car,carInfo)

                if car<self.cars-1:

                    car += 1
                    carInfo["inicio"] = carInfo["fin"]+1
                    carInfo["fin"] = index
                    carInfo = self.updateCar(sample,carInfo)

                else:

                    break

            self.carDist(sample,car,carInfo)


    def initPopulation(self):

        self.stage = 1

        for sample in range(self.n):

            index = 1
            self.Queue = self.rng.permutation(self.options)
            self.Tour[sample][:index] = np.copy(self.Queue[:index])
            self.subQueue = np.copy(self.Queue[index:])
            QueueLen = len(self.Queue)-1

            car = 0

            opW, weightW = self.weighting(self.Tour[sample][:index])
            
            carInfo = self.fillCar(0,index-1,opW,weightW)

            while index <= (QueueLen):

                availableWeight = self.flex["max"] - carInfo["peso"] 
                availableOp = self.oplimit - carInfo["op"] 
                posibles = self.filtering(self.subQueue,availableWeight,availableOp)
                
                if posibles.size == 0:

                    self.carDist(sample,car,carInfo)

                    if car<self.cars-1:

                        carInfo["inicio"] = carInfo["fin"]+1
                        car += 1
                        NextPoint = self.subQueue[0]
                        #Puede mejorar
                    else:

                        NextPoint = np.copy(self.subQueue)
                        self.Tour[sample][index:] = NextPoint
                        index += len(self.Tour[sample][index:])
                        break

                else:

                    reference = self.Tour[sample][index-1]
                    idNextPoint = self.NI(posibles,reference)
                    NextPoint = posibles[idNextPoint]
                
                carInfo["fin"] = index
                self.Tour[sample][index] = NextPoint
                self.removeInQ(NextPoint)

                carInfo = self.updateCar(sample,carInfo)
                self.carDist(sample,car,carInfo)
                index += 1


    def crossoverPopulation(self):

        self.stage = 2

        parentsIndex = self.bestFitness(self.nparents)
        self.TourParents =  self.Tour[parentsIndex]
        randomOrder = self.rng.permutation(parentsIndex)
        leng = len(randomOrder)//2
        couples = np.split(randomOrder,leng)

        for parentsId in range(len(couples)):
            parents = self.Tour[couples[parentsId]]
            self.TourChildren[parentsId] = self.crossover(parents)


    def joinPopulation(self):

        self.stage = 3

        self.oldTour = self.oldPopulation()
        self.Tour = np.concatenate((self.TourParents,self.TourChildren,self.oldTour),axis=0)
        self.rng.shuffle(self.Tour)


    def mutatePopulation(self):

        self.stage = 4

        candidates = self.rng.choice(self.Tour.shape[0],self.nmutation,replace=False)
        print(candidates)
        print("size Tour mutation")
        print(self.Tour.shape)
        for candidate in candidates:
            print(self.Tour[candidate])
            self.Tour[candidate] = self.mutate(self.Tour[candidate],self.noptions,0.2)


    def updateCarsCut(self):

        self.stage = 5

        self.CarCuts = np.empty([self.Tour.shape[0],self.cars],dtype=object)

        for sample in range(self.Tour.shape[0]):

            self.cutter(sample,self.Tour[sample])


    def orderingTrace(self):

        for sample in range(len(self.CarCuts)):
            for car in self.CarCuts[sample]:
                self.Tour[sample][car["inicio"]:car["fin"]+1] = self.orderFromCendis(self.Tour[sample][car["inicio"]:car["fin"]+1])


    def orderingTraceSample(self,CarCuts,Tour):

        for car in CarCuts:
            if car is not None:
                Tour[car["inicio"]:car["fin"]+1] = self.orderFromCendis(Tour[car["inicio"]:car["fin"]+1])
        
        return Tour
        


    def evolution(self,epochs):

        self.progress = 1.0/epochs
        self.initPopulation()
        self.updatePopFitness()
        self.means = np.array([self.fitnessMean()])
        for i in range(epochs):
            self.bar.progress(self.progress*(i+1))
            self.crossoverPopulation()
            self.joinPopulation()
            self.mutatePopulation()
            self.updateCarsCut()
            self.updatePopFitness()
            best = self.bestFitness(1)

            if self.popFitness[best] < self.bestOfBest["fitness"]:
                self.fillBest(self.popFitness[best],self.Tour[best],self.CarCuts[best],i)

            self.means = np.append(self.means,self.fitnessMean())
