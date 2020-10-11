# ProzhitoAutoencoder2Cluster

Description:  The following scripts are intended to allow the user to 1) select key root words in Russian, 2) search the Prozhito database (collection of Russian diaries from 1600-modern day) for instances of those words and pull the word vector representations of those words, 3) compress the word vectors using an autoencoder and 4) cluster the final product.   

# Installation
        
Notes: currently this repository is insufficient to run the first half of the code (selecting entries based on key stems and vectorizing them) It is missing both a) the sql connection/permissions to access the full database b) the russian language word vector model and c) several large files exceeding the 100 MB specification. Making these available for public use is still a work in progress.

# Files included:
* **keyword_vecs (will not work with current repository resources)**: this extracts relevant sentences and convert to word vectors
* **autoencoder_stikhi**:  this is an autoencoder set up used to compress the vector, written as a sort of psuedo function.  I found it easier to adjust the various parameters in this way as opposed to using a traditional function.
* **hdbscan_clusterer**:  this clusters the compressed vectors.  Since the HDBSCAN algorithm marks various points as noise, there is also an option to fit noise points retroactively with both a svm and rf classifier.  These are rather crude at the moment and do not have grid search etc built in to them.  Additionally, it is suggested to examine the clusters with and without noise, to make sure that fitting the noise points post hoc is a sensible thing to do
* **090120_autoencoderhelperfunctions**:  contains many relevant functions, including the one to run HDBSCAN + an ML classifier

# Current Requirements:
pandas, numpy, nltk, math, gensim, punctuation, Word2Vec(from gensim.models), sqlalchemy, punctuation(from string), sub (from re), tensorflow, keras, spatial(from scipy), HDBSCAN, sklearn , plotly, hiplot(optional), matplotlib



# MIT License

Copyright (c) 2020 Madeline Kehl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
