import numpy as np
import random
import copy
import math

from numpy import linalg as LA

from multiprocessing import Process, Array
from shared_ndarray import SharedNDArray



from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import canberra



N = 0
vocabulary = {}
vocabularyCount = {}
probsChoice = []
constWeight = 0


distances = [euclidean,canberra,cityblock,euclidean,cosine]


def  main(dataset,size=400,alpha=0.0275,window=3,negSampling=3,threshold=-1,eta=1,d=0,W=[],Importance=10,lamba=1.0,g=0):

    """
    Main Class For build word embedding DiVe 
    """
    
    global vocabulary
    global vocabularyCount
    
    countWords(dataset)
    
    train = []
    train = createVocabulary(dataset,window,threshold) 
    
    dataset = []
    
    createProbs(eta)
    
    qtd = len(train)/50
    init = 0
    end = qtd
    train = np.array(train)
    np.random.seed(0)

    outVector = np.zeros([len(vocabulary),size])
    
    if g == 0:
        inVector = np.random.uniform(low=-0.5/size,high=0.5/size,size=(len(vocabulary),size))
    else:
        inVector = np.random.uniform(low=-1,high=1,size=(len(vocabulary),size))

    train = SharedNDArray.copy(np.array(train))
    inVector = SharedNDArray.copy(inVector)
    outVector = SharedNDArray.copy(outVector)
    
    threads = []
    
    while end <= len(train.array):
        
        threads.append(Process(target=LearningEmbedding,args=(alpha,window,negSampling,size,eta,d,W,Importance,init,end,inVector,outVector,train,lamba)))
        init = end
        end += qtd
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    del qtd
    del train
    del init
    del end
    
    #vocabulary = {}
    vocabularyCount = {}
    return inVector,outVector,vocabulary

    

def countWords(dataset):

    '''
    
    Count Occurrence of Words in DataSet . Return Total Number of words apperrence 
    and total number apperrence of individual words 
    
    '''
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    
    
    N = 0
    vocabulary = {}
    vocabularyCount = {}
    probsChoice = []
    constWeight = 0

    for line in dataset:

        for word in line:

            if word not in vocabularyCount:

                vocabularyCount[word] = 0

            vocabularyCount[word] += 1

    for freqWord in vocabularyCount:

        N += vocabularyCount[freqWord]

def createVocabulary(dataset,window,threshold):

    '''
    Create the vocabulary using for train the model, if necessary discard words with low Occurrence.
    Build the train set based  n-grams(window) and words not words not discarded 
    '''
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    
    vocabulary = {}
    
    index = 0
    train = [] 
    
    for line in dataset:
        
        sequence = []

        for word in line:
            
            probs = vocabularyCount[word]
            
            if probs < threshold:
                continue
                
            if word not in vocabulary:
          
                vocabulary[word] = index
                index += 1

            sequence.append(vocabulary[word])
            
            if len(sequence) == window+1:

                sentence = []

                for i in range(0,window+1):

                    sentence.append(sequence[i])

                #mid = sentence.pop((window+1)/2)
                #sentence.append(mid)
                
                train.append(sentence)
                sequence.pop(0)

    return train



def createProbs(eta):

    '''
    Create a probability of apperrence of a word in dataset
    The probability will be use in negative Sampling and in Importance Sampling
    '''
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    
    
    N = 0 
    probsChoice = np.zeros(len(vocabulary))
    probsChoiceN = np.zeros(len(vocabulary))

    for freqWord in vocabularyCount:

        if freqWord in vocabulary:

                N += vocabularyCount[freqWord]


    for word in vocabulary:

        probsChoice[vocabulary[word]] = ((vocabularyCount[word])/float(N))**(3/4.0)
        constWeight += 1/float(probsChoice[vocabulary[word]])
        probsChoiceN[vocabulary[word]] = ((vocabularyCount[word])/float(N))**(3/4.0)
    
    norm = np.sum(probsChoice)
    probsChoice = probsChoice/float(norm)
    constWeight = constWeight**(-1)


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))
    
    
def mostSimilarI(w,d=0,k=25,outVector=[]):
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    
    global distances
    
    word = vocabulary[w]
    dist = []
    for y in vocabulary:

                    x = vocabulary[y]
                    rt = distances[d](outVector.array[word],outVector.array[x])
                    dist.append([rt,y])
    cont = 0
    counts = []
    for x in sorted(dist):

            print x
            #counts.append(C[vocabulary[x[1]]])
            cont += 1
            if cont == k:
                    break
    #return counts
    
    

def LearningEmbedding(alpha,window,negSampling,size,eta,d,W,Importance,init,end,inVector,outVector,train,lamba=0.0):

    '''
    Learning Dive Embedding 
    '''


    randC = len(train.array)
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    

    for c in range(init,end):


        negs = np.random.choice(len(vocabulary),negSampling,p=probsChoice)
        wordO = train.array[c][window]

        contexts = []

        for x in range(0,window):

            contexts.append(train.array[c][x])

        mvec = np.mean(inVector.array[contexts],axis=0)

        Dmvec = np.zeros(size)

        z = 0.5*(lamba*LA.norm(outVector.array[wordO])**2+lamba*LA.norm(mvec)**2-LA.norm(outVector.array[wordO]-mvec)**2)
        p = sigmoid(z)
        g = alpha *(1-p)

        Dmvec += g *(outVector.array[wordO]-mvec+lamba*mvec)

        outVector.array[wordO] += g*(-outVector.array[wordO]+mvec+lamba*outVector.array[wordO])

        for neg in range(0,negSampling):

            wordN = negs[neg]
            z = 0.5*(lamba*LA.norm(outVector.array[wordN])**2+lamba*LA.norm(mvec)**2-LA.norm(outVector.array[wordN]-mvec)**2)
            p = sigmoid(z)
            g = alpha * -p
            Dmvec += g *(outVector.array[wordN]-mvec+lamba*mvec)
            outVector.array[wordN] += g*(-outVector.array[wordN]+mvec+lamba*outVector.array[wordN])


        inVector.array[contexts] += Dmvec
        
        
        
def LearningEmbedding2(alpha,window,negSampling,size,eta,d,W,Importance,init,end,inVector,outVector,train,lamba=0.0):

    '''
    Learning Dive Embedding 
    '''


    randC = len(train.array)
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    

    for c in range(init,end):


        negs = np.random.choice(len(vocabulary),negSampling,p=probsChoice)
        wordO = train.array[c][window]

        contexts = []

        for x in range(0,window):

            contexts = train.array[c][x]

        mvec = np.mean(inVector.array[contexts],axis=0)

        Dmvec = np.zeros(size)

        z = 0.5*(lamba*LA.norm(outVector.array[wordO])**2+lamba*LA.norm(mvec)**2-LA.norm(outVector.array[wordO]-mvec)**2)
        #p = sigmoid(z)
        p = math.tanh(z)
        g = alpha *((1-p**2)/float(p))

        Dmvec += g *(outVector.array[wordO]-mvec+lamba*mvec)

        outVector.array[wordO] += g*(-outVector.array[wordO]+mvec+lamba*outVector.array[wordO])

        for neg in range(0,negSampling):

            wordN = negs[neg]
            
            z = 0.5*(lamba*LA.norm(outVector.array[wordN])**2+lamba*LA.norm(mvec)**2-LA.norm(outVector.array[wordN]-mvec)**2)
            top = 4*np.exp(2*z)
            down = (np.exp(2*z)+1)**2
            
            p = top/float(down)
            g = alpha * -p
            
            Dmvec += g *(outVector.array[wordN]-mvec+lamba*mvec)/float(math.tanh(-z))
            outVector.array[wordN] += g*(-outVector.array[wordN]+mvec+lamba*outVector.array[wordN])/float(math.tanh(-z))


        inVector.array[contexts] += Dmvec

        
        
        
def LearningEmbedding3(alpha,window,negSampling,size,eta,d,W,Importance,init,end,inVector,outVector,train,lamba=0.0):

    '''
    Learning Dive Embedding 
    '''


    randC = len(train.array)
    
    global N 
    global vocabulary
    global vocabularyCount
    global probsChoice
    global constWeight    

    for c in range(init,end):


        negs = np.random.choice(len(vocabulary),negSampling,p=probsChoice)
        wordO = train.array[c][window]

        contexts = []

        for x in range(0,window):

            contexts.append(train.array[c][x])

        mvec = np.mean(inVector.array[contexts],axis=0)

        Dmvec = np.zeros(size)

        #z = 0.5*(lamba*LA.norm(outVector.array[wordO])**2+lamba*LA.norm(mvec)**2-LA.norm(outVector.array[wordO]-mvec)**2)
        #p = sigmoid(z)
        #p = math.tanh(z)
        #g = alpha *((1-p**2)/float(p))

        Dmvec += alpha *(outVector.array[wordO]-mvec+lamba*mvec)

        outVector.array[wordO] += alpha*(-outVector.array[wordO]+mvec+lamba*outVector.array[wordO])

        for neg in range(0,negSampling):

            wordN = negs[neg]
            
            #z = 0.5*(lamba*LA.norm(outVector.array[wordN])**2+lamba*LA.norm(mvec)**2-LA.norm(outVector.array[wordN]-mvec)**2)
            #top = 4*np.exp(2*z)
            #down = (np.exp(2*z)+1)**2
            
            #p = top/float(down)
            #g = alpha * -p
            
            Dmvec += alpha *(outVector.array[wordN]-mvec+lamba*mvec)
            outVector.array[wordN] += alpha*(-outVector.array[wordN]+mvec+lamba*outVector.array[wordN])


        for n in contexts:
            
            inVector.array[n] += Dmvec

            
            
import sys

if __name__ == '__main__':
    
    
    print sys.argv
    
    fh = open(sys.argv[1],'r')
    sentences = []
    for line in fh:
        #print line
        aux = []
        sep = line.split(" ")
        for y in range(0,len(sep)-1):
            
            aux.append(sep[y])
            
        sentences.append(aux)
        
    fh.close()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #sentences = [x.strip() for x in content] 

    sentences = np.array(sentences)
    
    print sentences[0]
    vec,out,voc = main(dataset=sentences,size=400,window=5,d=0,lamba=1.0)
    
    fi = open(sys.argv[2],"w+")
    cont =0
    
    for word in voc:
        cont+=1
        fi.write(str(word)+" ")
        for numbers in vec.array[voc[word]]:
            
            fi.write(str(numbers)+" ")
            
        fi.write("\n")
        
    fi.close()
