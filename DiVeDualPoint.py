import numpy as np
import pandas as pd
import random
import copy
import math

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import canberra

class DSWEwordEmbedding:

    N = 0

    def __init__(self,dataset,size=400,alpha=0.25,window=5,negSampling=4,threshold=-1,eta=0,d=0,W=[]):

        #### Word Count and Probability 
        self.N = 0
        self.vocabulary = {}
        self.vocabularyCount = {}
        self.probsChoice = []
        self.constWeight = 0


        self.train = []
        self.invector = []
        #self.bias = []

        self.alpha = alpha
        self.window = window

        self.distances = [euclidean,canberra,cityblock,euclidean,cosine]

        #Cont Occurrence of Words
        self.countWords(dataset)
        self.createVocabulary(dataset,window,threshold) 

        # Importance Sampling 
        self.createProbs(eta)

        self.LearningEmbedding(alpha,window,negSampling,size,eta,d,W)

    def countWords(self,dataset):

        for line in dataset:

            for word in line:

                if word not in self.vocabularyCount:

                    self.vocabularyCount[word] = 0

                self.vocabularyCount[word] += 1

        for freqWord in self.vocabularyCount:

            self.N += self.vocabularyCount[freqWord]

    def mostSimilarI(self,w,d=0,k=25):

        #np.sqrt(self.inVector)
        word = self.vocabulary[w]
        dist = []
        for y in self.vocabulary:

                        x = self.vocabulary[y]
                        rt = self.distances[d](self.inVector[word],self.inVector[x])
                        dist.append([rt,y])
        cont = 0
        for x in sorted(dist):

                print x
                cont += 1
                if cont == k:
                        break  

    def createVocabulary(self,dataset,ws,threshold):

        index = 0

        for line in dataset:

            sequence = []
            auxCosine = []

            window = ws

            for word in line:

                probs = self.vocabularyCount[word]/float(self.N)


                if probs < threshold:
                    continue

                if word not in self.vocabulary:
              
                    self.vocabulary[word] = index
                    index += 1

                sequence.append(self.vocabulary[word])

                if len(sequence) == window+1:

                   
                    sentence = []

                    for i in range(0,window+1):

                        sentence.append(sequence[i])

                    self.train.append(sentence)
                    sequence.pop(0)

    def createProbs(self,eta):

        self.N = 0
        self.probsChoice = np.zeros(len(self.vocabulary))
        self.probsChoiceN = np.zeros(len(self.vocabulary))

        for freqWord in self.vocabularyCount:

            if freqWord in self.vocabulary:

                    self.N += self.vocabularyCount[freqWord]


        for word in self.vocabulary:

            self.probsChoice[self.vocabulary[word]] = ((self.vocabularyCount[word])/float(self.N))**(eta)
            self.constWeight += 1/float(self.probsChoice[self.vocabulary[word]])
            self.probsChoiceN[self.vocabulary[word]] = ((self.vocabularyCount[word])/float(self.N))**(1)
        
        norm = np.sum(self.probsChoice)
        self.probsChoice = self.probsChoice/float(norm)
        #self.constWeight = 1 /self.constWeight
        
    def gradient(self,X,Y,d,context=-1,word=-1):

        if (X==Y).all():

            return 0,0

        if d == 0:

            context = (X-Y)/self.distances[d](X,Y)
            word = -(X-Y)/self.distances[d](X,Y)


            return context,word

        if d == 1:

            top = (X-Y)*(np.abs(Y)*np.abs(X)+X*Y)
            down = np.abs(X)*(np.abs(X)+np.abs(Y)**2)*np.abs(X-Y)

            topW = (Y-X)*(np.abs(Y)*np.abs(X)+X*Y)
            downW = np.abs(Y)*(np.abs(X)+np.abs(Y)**2)*np.abs(X-Y)

            return (top/down),(topW/downW)

        if d == 2:

            context = (X-Y)/(np.abs(X-Y))
            word = (Y-X)/np.abs(X-Y)

            return  context,word
        #Need to build
        if d == 3:

            context = (X-Y)
            word = (Y-X)

            return  context,word

    def LearningEmbedding(self,alpha,window,negSampling,size,eta,d,W):

        B = np.zeros(len(self.vocabulary))

        randC = len(self.train)
        randV = len(self.vocabulary)


        if len(W) == 0:

            if d  == 0 or d == 2 :

                self.inVector = np.random.uniform(low=-1,high=1,size=(len(self.vocabulary),size))

            elif d == 1:
                
                self.inVector = np.random.uniform(low=5,high=15,size=(len(self.vocabulary),size))

            else:

                self.inVector = np.random.uniform(low=-0.5/size,high=0.5/size,size=(len(self.vocabulary),size))

        else:
                self.inVector = W

        
        self.outVector = np.zeros([len(self.vocabulary),size])

        partition = 1
        j = 0
        cont = 0
        if len(self.train) > 200000:

            self.alpha = 0.01

        for i in range(0,len(self.train)):
          
            j = (j% len(self.vocabulary)) + 1
         
            if j >= len(self.vocabulary):

                j = 0

            if i % 10000 == 0:

                alpha = self.alpha * (1 - float(i) / len(self.train))

                if alpha < self.alpha * 0.0001:

                    alpha = self.alpha * 0.0001

            c = i

            wordV = np.random.choice(randV,1,p=self.probsChoice)[0]

            wordO = self.train[c][window]

            negs = np.random.choice(randV,negSampling,p=self.probsChoiceN)

            dist = 0
            contexts = []

            for x in range(0,window):
                contexts.append(self.train[c][x])

            mvec = np.mean(self.inVector[contexts],axis=0)
            dist += -(self.distances[d](mvec,self.outVector[wordV]))
            s = np.exp(dist)

            partition = partition + s - B[j]

            B[j] = s

            PropCO = s/float(partition)

            GradWord = self.gradient(mvec,self.outVector[wordO],d,mvec,wordO)[1]
            self.outVector[wordO] -= alpha*np.exp(-euclidean(mvec,self.outVector[wordO]))*GradWord

            GradWord = self.gradient(mvec,self.outVector[wordV],d,mvec,wordV)[1]
            self.outVector[wordV] += alpha*GradWord*PropCO*self.constWeight

            for neg in range(0,negSampling):


                wordN = negs[neg]
                
                GradWord = self.gradient(mvec,self.outVector[wordN],d,mvec,wordN)[1]
                self.outVector[wordN] += alpha*np.exp(-euclidean(mvec,self.outVector[wordN]))*GradWord

            for n in contexts:

                if n != wordO:
                    GradCon = self.gradient(self.inVector[n],self.outVector[wordO],d,self.inVector[n],wordO)[0]
                    self.inVector[n] -= alpha*np.exp(-euclidean(self.inVector[n],self.outVector[wordO]))*GradCon
                
                if n!= wordV:
                    GradCon = self.gradient(self.inVector[n],self.outVector[wordV],d,self.inVector[n],wordO)[0]
                    self.inVector[n] += alpha*GradCon*PropCO*self.constWeight

            cont += 1





if __name__ == '__main__':
    
    print "Hello World"
