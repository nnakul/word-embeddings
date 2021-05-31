## Word Embeddings
*Word embedding* is used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning. Word embeddings can be obtained using a set of language modeling and feature learning techniques where words or phrases from the vocabulary are mapped to vectors of real numbers. This project is mainly inspired from the research work of *Mikolov et al.* published in the paper <a href="https://arxiv.org/pdf/1301.3781.pdf"> *Efficient Estimation of Word Representations in Vector Space* </a>. In the paper, the authors have proposed two novel model architectures for computing continuous vector representations of words from very large data sets, *Continuous Bag-of-Words Model* and *Continuous Skip-gram Model*. This project uses the *Continuous Skip-gram Model* to learn word embeddings from an *11 million* words text corpus, having a vocabulary size of *202,000* (without re-sampling and filtering).<br> In conjunction with the basics discussed in this paper, the project also implements the advanced sampling techniques and extensions of the Continuous Skip-gram Model presented in another paper of *Mikolov et al.*, <a href="https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf"> *Distributed Representations of Words and Phrases and their Compositionality* </a>. These extensions improve both the quality of the vectors and the training speed. By subsampling of the frequent words, significant speedup is obtained and also more regular word representations are learned.

## Negative Sampling
*Negative Sampling (NEG)* can be seen as an approximation to *Noise Contrastive Estimation (NCE)*. *NCE* approximates the loss of the softmax as the number of samples (or target classes) increases. *NEG* simplifies *NCE* and does away with this guarantee, as the objective of *NEG* is to learn high-quality word representations rather than achieving low perplexity on a test set, as is the goal in language modelling. <br>
*NEG* uses a logistic loss function to minimise the negative log-likelihood of words in the training set. The task is to use logistic regression in order to distinguish the true target from a *k*-size subset of all possible targets, the subset being constructed considering a noise distribution over all the targets. The targets in this *k*-subset are called *negative samples* and the noise distribution over all the samples is based on their frequency in the training corpus (as described in the paper). In this project, 10 negative samples are chosen for every training data sample (*k = 10*).

## Training
The model was trained over *9 epochs*, with a learning rate of *0.003*.<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120228304-79283e80-c268-11eb-88bc-a51ff90f2fa0.png" width = '400' height = '320'> 

## Performance

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120229946-e7bacb80-c26b-11eb-9114-7fe61bbb55ff.png" width = '400' height = '320'>

## Analogical Reasoning

## Improvements
