from glove import Corpus, Glove
from gensim.models import Word2Vec

import numpy as np
import scipy as sp
import scipy.stats
import fasttext
import pandas as pd
import os 

os.environ["THEANO_FLAGS"] = "floatX=float32"
os.environ["THEANO_FLAGS"] = 'optimizer=fast_compile, floatX=float32,exception_verbosity=high'

from interfaces.interface_configurator import InterfaceConfigurator
from libraries.evaluation.support import evaluate
from libraries.evaluation.lexsub.run_lexsub import run_lexsub



def gloveData(sentences,glove,sizeEmb,window,min_count=1,test=0):

    X = []

    if test == 0:

        glove = Glove(no_components=sizeEmb)
        corpus = Corpus()
        corpus.fit(sentences, window=window)

        glove.fit(corpus.matrix)
        glove.add_dictionary(corpus.dictionary)

        for x in sentences:

            aux = np.zeros(sizeEmb)

            for word in x:

                if word in glove.dictionary:

                    aux = aux+glove.word_vectors[glove.dictionary[word]]

            X.append(aux)
    else:

        for x in sentences:

            aux = np.zeros(sizeEmb)

            for word in x:

                if word in glove.dictionary:

                    aux = aux+glove.word_vectors[glove.dictionary[word]]

            X.append(aux)

    return X,glove

def word2vecData(sentences,word2vec,sizeEmb,window,min_count=1,test=0):

    X= []

    if test == 0:

        word2vec = Word2Vec(sentences, size=sizeEmb, window=window, min_count=min_count)

        for x in sentences:

            aux = np.zeros(sizeEmb)

            for word in x:

                if word in word2vec:

                    aux = aux+word2vec[word]

            X.append(aux)
    else:

        for x in sentences:

            aux = np.zeros(sizeEmb)

            for word in x:

                if word in word2vec:

                    aux = aux+word2vec[word]

            X.append(aux)

    return X,word2vec


def FastText(sentences,word2vec,sizeEmb,window,min_count=1,test=0):

    X = []

    if test == 0:

        joinData = []

        for x in sentences:
            
            aux = ""

            for word in x:

                aux += word+" "

            joinData.append(aux)


        outfile = open("data.txt","w") 
        outfile.write("\n".join(joinData))

        word2vec = fasttext.cbow("data.txt", 'model',dim=sizeEmb,ws=window)

        for x in sentences:

            aux = np.zeros(sizeEmb)

            for word in x:

                if word in word2vec:

                    aux = aux+word2vec[word]

            X.append(aux)
    else:

        for x in sentences:

            aux = np.zeros(sizeEmb)

            for word in x:

                if word in word2vec:

                    aux = aux+word2vec[word]

            X.append(aux)

    return X,word2vec

def buildBayesianSG(sentences,bg,sizeEmb,window,min_count=1,test=0):

    joinData = []
    X = []

    for x in sentences:
    
        aux = ""

        for word in x:

            aux+=word+" "

        joinData.append(aux)


    if test == 0:

        outfile = open("data.txt","w") 
        outfile.write("\n".join(joinData))

        train_data_path = 'data.txt'
        vocab_file_path = 'b.txt' 
        output_folder_path = ""  

        i_model = InterfaceConfigurator.get_interface(train_data_path, vocab_file_path, output_folder_path)

        i_model.train_workflow()

        # store the temporary vocab, because it can be different from the original one(e.g. smaller number of words)
        vocab = i_model.vocab

        temp_vocab_file_path = os.path.join(i_model.output_path, "vocab.txt")
        vocab.write(temp_vocab_file_path)

        mu_vecs = [os.path.join(i_model.output_path, "mu.vectors")]
        sigma_vecs = [os.path.join(i_model.output_path, "sigma.vectors")]

        vec = open(mu_vecs[0],"r")
        m = {}
        inVector = []
        cont = 0
        for line in vec:

                so = line.split(" ")
                m[so[0]] = cont
                aux = so[1:]
                inVector.append(aux)

        inVector = np.array(inVector).astype('float32')

        for line in sentences:

            aux = np.zeros(sizeEmb)

            for word in line:

                if word in m:

                    aux += inVector[m[word]]

            X.append(aux)

        return X,[inVector,m]

    else:

        for line in sentences:

            aux = np.zeros(sizeEmb)

            for word in line:
                
                if word in bg[1]:

                    aux += bg[0][bg[1][word]]

            X.append(aux)
        return X,[bg[0],bg[1]]

if __name__ == '__main__':
    
    print "Hello World"




