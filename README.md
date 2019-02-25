#                          DiVe Word Embedding
authors are removed for double blind review 
***

### Requirements
1. Python 2.7
2. Numpy 1.65
3. Pandas 0.24

### Input Data
DiVe receive as input any sequence of strings(sentences), each string of trainning corpus will be map to a vector.
Each file should have one sentence per line as follows (space delimited): \
`weaknesses minor feel layout remote control n show complete file names mp3s.`\
`normal size sorry ignorant way get back 1x quickly.` \
`many disney movies n play dvd player` \
`...`

### Training DiVe
For training DiVe you need choose a model a type the follow command:\
` python DiVeDualPoint.py data.txt wordsVectors.out`\
wordsVectors.out will be the output, each word in vocabulary represents a line and its coordenates in the embedding, as:
`house -1.0 2.4 -0.3 ... ` \
`car 1.5 0.01 -0.2 -1.1 ...`

### Reference

Please make sure to cite the papers when its use for represents word similarity by word embedding.

Please cite the following paper if you use this implementation:\
`
@InProceedings{XxX,`\
  `author    = {XxX, YyY, ZzZ, SsS, DdD, EeE},`\
  `title     = {DiVe: Distance based Vectors Embedding technique for effective text classification},`\
  `booktitle = {Proceedings of ACL},`\
  `year      = {2019} }`
