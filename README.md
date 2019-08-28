#                          DiVe Word Embedding
authors are removed for double blind review 
***

In this repository we present DiVe (Distance-based Vector Embedding), a new word embedding technique based on a scalable Markovian statistical model to represent sequences of words. Our experiments demonstrate that DiVe is able to outperform existing, more complex, machine learning approaches, while preserving simplicity and scalability.

### Requirements
1. Python 2.7
2. Numpy 1.65
3. Pandas 0.24


### Datasets already used for text representation 

|name | task | vocabulary | size | classes  |
|----------	|------------------------------	|-----------:|----------:|:-----------:|
|[CR](https://github.com/davidsbatista/Aspect-Based-Sentiment-Analysis/tree/master/datasets/CR)  | User review polarity | 5176 | small | 2 |
|[HSTW](https://github.com/zeerakw/hatespeech)  | Hate speech detect| 23739 | large |3  |
|[PO](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)| Sentece polarity | 18179| large |  2 |
|[SIM](https://github.com/hallr/DAT_SF_19/blob/master/data/yelp_labelled.txt)  | Movie and TV Review | 1000 | small|  2|
|[YR](https://github.com/hallr/DAT_SF_19/blob/master/data/yelp_labelled.txt)  | Food review polarity | 1000| small| 2|
|[SUBJ](http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz) | Subjectivity and objectivity | 18179 | large | 2 |
|[AR](https://github.com/hallr/DAT_SF_19/blob/master/data/yelp_labelled.txt)  | User product review  | 1000| small | 2  |
|[QTS](http://cogcomp.org/Data/QA/QC/)  | Question Answering  | 16504 | small | 6  |
|[IM](https://drive.google.com/file/d/0B8yp1gOBCztyN0JaMDVoeXhHWm8/)  | Movie and TV Review | 74337| large| 2 | 

### Input Data
DiVe receive as input any sequence of strings(sentences), each string of trainning corpus will be map to a vector.
Each file should have one sentence per line as follows (space delimited): \
`...`\
`weaknesses minor feel layout remote control show complete file names mp3s`\
`normal size sorry ignorant way get back 1x quickly` \
`many disney movies n play dvd player` \
`...`

### Training DiVe
For training DiVe you need choose a model a type the follow command:\

` python DualPoint.py data.txt wordsVectors.out`\
wordsVectors.out will be the output, each word in vocabulary represents a line and its coordenates in the embedding, as:
`house -1.0 2.4 -0.3 ... ` \
`car 1.5 0.01 -0.2 -1.1 ...`

###  Analysis of parameter α in similarity function f

In our work, we demonstrate that α can easily change the model accuracy, as follows,

![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/go.png "Title")

we compare results of DiVe Single and Dual models. In both cases we observe a large variation in terms of F1 depending on α. For example, for the QTS dataset, the F1 score has almost 30% variation for the DiVe Dual Point model, and 10.5\% variation for the Single Point model, and for SUBJ dataset 18\% for Dual Point and almost 20% for Single Point. This shows that α can significantly influence an estimator's accuracy, therefore, this results which suggests that it is worth setting this hyperparameter using cross-validation instead of keeping it fixed.


Inner product Embedding|  Euclidean Embedding|
:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/cosine.png  "Title") |  ![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/euclidean.png  "Title")


In addition, to visualize the embedding when is training with α set as 0 or 1, we design a simple experiment to evaluate how words with high/low individual occurrences are grouped in embedding. Therefore, we train DiVe with sigmoid activation in CR dataset, we plot every word vector with reduced dimensions by TSNE technique. The results is showed in figures above. We mark for high occurrences the top 15% of most occurring words in this dataset, for medium top 25%, for low the rest. We observe, in Inner Product Embedding, that the words with high occurrence, the two portions in right, are grouped together, and words with medium and low occurrence are surrounded this two portions. However, in figure,Euclidean Embedding, the two portions of high occurrence are separated in graph, moreover, in the middle are majority occupied by medium occurrence and the low occurrence words are surrounded the other words.


###  Performance of classifiers with trained embeddings
We compare the quality of the embeddings obtained with DiVe to the following word embeddings baseline techniques: Word2Vec, Glove, Bayesian SkipGram and FastText. The embeddings were trained on the specific dataset whose sentences we want to classify.

IM             |  CR|  HSTW
:-------------------------:|:-------------------------:|:-------------------------:
![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/polarity(1).png  "Title") |  ![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/polarity(1).png  "Title")|![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/polarity(1).png "Title")
PO             |  YR|  QS
![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/polarity(1).png "Title") |  ![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/polarity(1).png "Title")|![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/question(1).png "Title")

Our results showed that DiVe overcome four popular word embedding, namely Word2Vec, Glove, FastText e Bayesian Skip Gram, and more than 9 datasets.

In addition, we conducted a hypothesis test on whether DiVe’s model is in fact different from others, based on McNemar’s test to know more about this test check https://towardsdatascience.com/statistical-tests-for-comparing-machine-learning-and-baseline-performance-4dfc9402e46f and https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/. In next figures we show the heat maps of p-values for all combinations of DiVe and classifiers and Word2Vec or Glove and classifiers, respectively, in the PO
dataset. Comparing the two heat maps, we observe that there are a lot of values < 0.05 that deny the null hypothesis, besides that, we plot the p-values from McNemar’s test as a CDF for all word embeddings, separated by dataset. We
can see that 2 out of 9 datasets (AR and YR) have < 40% of p-values < 0.05. In the remaining datasets, there are ≥ 60%
of p-values < 0.05.

Glove             |  Word2Vec|  All baselines for all datasets
:-------------------------:|:-------------------------:|:-------------------------:
![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/heatglove.png  "Title") |  ![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/heatw2v.png  "Title")|![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/cdfs.png "Title")


###  Performance of classifiers with pre-trained embeddings

We also evaluate results from deep learning techniques ELMo and BERT. We used these baselines as pre-trained embeddings, they were trained on a large dataset (Wikipedia and BookCorpus) and used to classification task. These techniques represent the state-of-the-art techniques for several NLP tasks. We can see that DiVe, even without tuning the f function,
outperformed ELMo in 4 classification tasks (YR, HSTW, AR,
and CR), and outperformed BERT in 3 classification tasks
(SUBJ, HSTW and YR)


![Figure 1 ](https://github.com/DiVeWord/DiVeWordEmbedding/blob/master/figs/Goes2.png "Title") 


In this work we introduced DiVe (Distance-based Vector Embedding), a new word embedding technique based on a
scalable Markovian statistical model to represent sequences of words. Our experiments showed that DiVe is a scalable
model for representing word sequences. One of the main building blocks od DiVe is an efficient algorithm to estimate
the partition function, which has been the main limitation of embedding techniques based on Markovian statistical models. In this work we used the F 1 score to evaluate the general quality of the word embeddings produced by DiVe in text classification tasks. In future work, we will analyze other metrics to capture how well human-perceived similarities and opposite-polarity words are represented by DiVe. We also plan to enhance the algorithm with context aware information, such as sentiment-related tagging.

### Reference

Please make sure to cite the papers when its use for represents word similarity by word embedding.

Please cite the following paper if you use this implementation:\
`
@InProceedings{?,`\
  `author    = {removed for double blind review},`\
  `title     = {DiVe: Distance based Vectors Embedding technique for effective text classification},`\
  `booktitle = {WSDM'20},`\
  `year      = {2020} }`
