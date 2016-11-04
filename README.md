# incremental-sequence-learning
Implementation of the Incremental Sequence Learning algorithms described in the Incremental Sequence Learning article

#Requirements
Python 3.5
Tensorflow 0.9


#Getting started
Parameter files for the first 3 experiments described in the article are available as exp/exp1a..d, exp/exp2a..d, and exp/exp3a..d. The a, b, c, and d variant represent the four different configurations compared in the article.

To start a run for experiment 1a, use:
./runrnn exp1a --runnr 1


#Data
This project makes use of the MNIST stroke sequence data set, available here:

https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data


#Results

I have included the R scripts used to extract results from the output files. To process the results, you can use:

source('R/process.R')
source('R/processruns.R')

binsize = 1000
requiredfraction = .9 #fraction of the files required to be available for reporting output
windowsize = 1
folder = '~/code/digits/rnn'

exp1atrain = processruns( 'exp1a', 'train', 1, binsize, windowsize, folder, requiredfraction )
exp1atest  = processruns( 'exp1a', 'test',  1, binsize, windowsize, folder, requiredfraction )

#Acknowledgements

The network architecture used in this work is based on the article [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf) by Alex Graves.

The implementation is based on the [write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow) by [hardmaru](https://github.com/hardmaru), which in turn is based on
the [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) implementation by [sherjilozair](https://github.com/sherjilozair). See the blog post [Handwriting Generation Demo in TensorFlow](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/).

