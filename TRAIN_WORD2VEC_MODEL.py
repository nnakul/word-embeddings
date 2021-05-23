import torch
import nltk
import pickle
from math import *
import numpy as np
from time import time
from random import random
from torch import nn as nn
from collections import Counter

TRAIN_CORPUS = ""
VOCABULARY = set()
VOCAB_SIZE = 0
SENTENCE_TOKENS = list()
WORD_TOKENS = list()
WORD_TO_ID = dict()
ID_TO_WORD = list()
CONTEXT_WINDOW = 5
SYMBOLS = "abcdefghijklmnopqrstuvwxyz0123456789-"
FREQUENCY_DIST_TABLE = dict()
WORD_IDX_TO_SENT_IDX = dict()
SENT_IDX_TO_WORD_IDX = dict()
INPUT_EMBEDDING_MATRIX = None
OUTPUT_EMBEDDING_MATRIX = None
OPTIMIZER_INPUT = None
OPTIMIZER_OUTPUT = None
DEVICE = torch.device("cuda")
CONTEXT_WINDOW = 5
EMBEDDING_DIMENSION = 300
TOTAL_EPOCHS = 9
LEARNING_RATE = 0.003
NOISE_DISTRIBUTION = None
FREQUENCY_LOWER_LIMIT = 6
BATCH_SIZE = 512
LOSS_PROGRESS = list()

def LoadTrainCorpus ( path ) :
    start_time = time()
    print('\n    [ LOADING TRAINING CORPUS ... ]')
    global TRAIN_CORPUS
    text_file = open(path, 'r', encoding="utf8")
    TRAIN_CORPUS = text_file.read()
    text_file.close()
    print('       [ CORPUS LOADED IN {} SECS ]'.format(round(time()-start_time, 3)))

def IsAlphabetic ( word ) :
    for symbol in word :
        if symbol in SYMBOLS[:26] : return True
    return False 

def RefineToken ( token ) :
    refined_token = token
    while refined_token[0] not in SYMBOLS[:-1] :
        refined_token = refined_token[1:]
    while refined_token[-1] not in SYMBOLS[:-1] :
        refined_token = refined_token[:-1]
    return refined_token

def RemoveLessFrequentWords ( ) :
    global SENTENCE_TOKENS, VOCABULARY, WORD_TOKENS, FREQUENCY_DIST_TABLE, VOCAB_SIZE
    frequency_distribution = list(Counter(WORD_TOKENS).items())
    frequency_distribution.sort(key = lambda x: x[1])
    noisy_words = set()
    for word, freq in frequency_distribution :
        if freq >= FREQUENCY_LOWER_LIMIT : break
        noisy_words.add(word)
    VOCABULARY -= noisy_words

    SENTENCE_TOKENS_old = SENTENCE_TOKENS
    SENTENCE_TOKENS = list()
    WORD_TOKENS = list()
    
    for sentence in SENTENCE_TOKENS_old :
        filtered_words = list()
        for word in sentence :
            if not word in noisy_words :
                filtered_words.append(word)
                WORD_TOKENS.append(word)
        SENTENCE_TOKENS.append(filtered_words)
    
    FREQUENCY_DIST_TABLE = Counter(WORD_TOKENS)
    VOCAB_SIZE = len(VOCABULARY)

def ProcessTrainCorpus ( ) :
    start_time1 = time()
    print('\n    [ PRE-PROCESSING TRAINING CORPUS ... ]')
    global SENTENCE_TOKENS, TRAIN_CORPUS, VOCABULARY, VOCAB_SIZE, WORD_TOKENS
    SENTENCE_TOKENS = list()
    WORD_TOKENS = list()
    VOCABULARY = set()
    TRAIN_CORPUS = TRAIN_CORPUS.lower()
    start_time = time()
    print('       [ TOKENIZING & REFINING TEXT ... ]')
    sentence_tokens = nltk.tokenize.sent_tokenize(TRAIN_CORPUS)
    for sentence in sentence_tokens :
        word_tokens = nltk.tokenize.word_tokenize(sentence)
        refined_word_tokens = list()
        for token in word_tokens :
            if ( IsAlphabetic(token) ) :
                refined_token = RefineToken(token)
                refined_word_tokens.append(refined_token)
                VOCABULARY.add(refined_token)
                WORD_TOKENS.append(refined_token)
        SENTENCE_TOKENS.append(refined_word_tokens)
    print('          [ TEXT TOKENIZED IN {} SECS ]'.format(round(time()-start_time, 3)))
    start_time = time()
    print('       [ FILTERING NOISE ... ]')
    RemoveLessFrequentWords()
    print('          [ NOISE FILTERED IN {} SECS ]'.format(round(time()-start_time, 3)))
    start_time = time()
    print('       [ SUB-SAMPLING TOKENS ... ]')
    SubSampling()
    print('          [ SUB-SAMPLING COMPLETED IN {} SECS ]'.format(round(time()-start_time, 3)))
    print('       [ PRE-PROCESSING COMPLETED IN {} SECS ]'.format(round(time()-start_time1, 3)))

def SubSampling ( ) :
    global SENTENCE_TOKENS, VOCABULARY, WORD_TOKENS, FREQUENCY_DIST_TABLE, VOCAB_SIZE
    THRESHOLD_PARAMETER = 1e-5 * len(WORD_TOKENS)
    prob_drop = {word : 1 - sqrt(THRESHOLD_PARAMETER/FREQUENCY_DIST_TABLE[word]) for word in VOCABULARY}
    
    SENTENCE_TOKENS_old = SENTENCE_TOKENS
    SENTENCE_TOKENS = list()
    WORD_TOKENS = list()
    
    for sentence in SENTENCE_TOKENS_old :
        filtered_words = list()
        for word in sentence :
            if random() >= prob_drop[word] :
                filtered_words.append(word)
                WORD_TOKENS.append(word)
        if ( len(filtered_words) > 0 ) :
            SENTENCE_TOKENS.append(filtered_words)
    
    VOCABULARY = set(WORD_TOKENS)
    VOCAB_SIZE = len(VOCABULARY)
    FREQUENCY_DIST_TABLE = Counter(WORD_TOKENS)

def CreateLookupTables ( ) :
    start_time = time()
    print('\n    [ CREATING LOOK-UP TABLES ... ]')
    global WORD_TO_ID, ID_TO_WORD, WORD_IDX_TO_SENT_IDX, SENT_IDX_TO_WORD_IDX
    WORD_TO_ID = dict()
    
    ID_TO_WORD = list(FREQUENCY_DIST_TABLE.items())
    ID_TO_WORD.sort(key = lambda x: x[1])
    ID_TO_WORD = [ word for word, freq in ID_TO_WORD ][::-1]
    
    for idx, word in enumerate(ID_TO_WORD) :
        WORD_TO_ID[word] = idx
    
    WORD_IDX_TO_SENT_IDX = dict()
    SENT_IDX_TO_WORD_IDX = dict()

    total_words_seen = 0
    for sent_idx, sentence in enumerate(SENTENCE_TOKENS) :
        temp_dict = {word_idx: sent_idx for word_idx in range(total_words_seen, total_words_seen+len(sentence))}
        SENT_IDX_TO_WORD_IDX[sent_idx] = (total_words_seen, total_words_seen+len(sentence)-1)
        WORD_IDX_TO_SENT_IDX.update(temp_dict)
        total_words_seen += len(sentence)
    
    print('       [ LOOKUP-UP TABLES CREATED IN {} SECS ]'.format(round(time()-start_time, 3)))

def GetTarget ( central_idx ) :
    window = np.random.randint(1, CONTEXT_WINDOW + 1)
    lower, upper = SENT_IDX_TO_WORD_IDX[WORD_IDX_TO_SENT_IDX[central_idx]]
    start_idx = central_idx - window if (central_idx - window) >= lower else lower
    end_idx = central_idx + window if (central_idx + window) <= upper else upper
    target_words = WORD_TOKENS[start_idx:central_idx] + WORD_TOKENS[central_idx+1:end_idx+1]
    return [ WORD_TO_ID[word] for word in target_words ]

def GetBatches ( batch_size ) :
    batch_count = len(WORD_TOKENS) // batch_size
    word_tokens_sub = WORD_TOKENS[:batch_count*batch_size]
    word_tokens_sub = list(enumerate(word_tokens_sub))
    SAMPLES = list()
    for idx in range(0, len(word_tokens_sub), batch_size) :
        x, y = list(), list()
        batch = word_tokens_sub[idx:idx+batch_size]
        for word_idx, word in batch :
            batch_x = WORD_TO_ID[WORD_TOKENS[word_idx]]
            batch_y = GetTarget(word_idx)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        SAMPLES.append((x, y))
    return SAMPLES

def InitializeNoiseDistribution ( ) :
    start_time = time()
    print('\n    [ INITIALIZING NOISE DISTRIBUTION ... ]')
    global NOISE_DISTRIBUTION
    word_freqs = np.array(sorted(FREQUENCY_DIST_TABLE.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    NOISE_DISTRIBUTION = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))
    print('       [ NOISE DISTRIBUTION INITIALIZED IN {} SECS ]'.format(round(time()-start_time, 3)))

def GetForwardNoise ( batch_size , sample_count ) :
    noise_words = torch.multinomial(NOISE_DISTRIBUTION, batch_size * sample_count, replacement=True)
    noise_words = noise_words.to(DEVICE)
    noise_vector = OUTPUT_EMBEDDING_MATRIX(noise_words).view(batch_size, sample_count, EMBEDDING_DIMENSION)        
    return noise_vector

def GetForwardLoss ( input_tensor , output_tensor , noise_tensor ) :
    batch_size, embed_size = input_tensor.shape
    input_tensor = input_tensor.view(batch_size, embed_size, 1)
    output_tensor = output_tensor.view(batch_size, 1, embed_size)
    out_loss = torch.bmm(output_tensor, input_tensor).sigmoid().log()
    out_loss = out_loss.squeeze()
    noise_loss = torch.bmm(noise_tensor.neg(), input_tensor).sigmoid().log()
    noise_loss = noise_loss.squeeze().sum(1)
    return -(out_loss + noise_loss).mean()

def InitializeWeightsAndOptimizers ( ) :
    start_time = time()
    print('\n    [ INITIALIZING EMBEDDING WEIGHTS ... ]')
    global INPUT_EMBEDDING_MATRIX, OUTPUT_EMBEDDING_MATRIX, OPTIMIZER_INPUT, OPTIMIZER_OUTPUT
    INPUT_EMBEDDING_MATRIX = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION).to(DEVICE)
    OUTPUT_EMBEDDING_MATRIX = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION).to(DEVICE)
    INPUT_EMBEDDING_MATRIX.weight.data.uniform_(-1,1)
    OUTPUT_EMBEDDING_MATRIX.weight.data.uniform_(-1,1)
    OPTIMIZER_INPUT = torch.optim.Adam(INPUT_EMBEDDING_MATRIX.parameters(), lr = LEARNING_RATE)
    OPTIMIZER_OUTPUT = torch.optim.Adam(OUTPUT_EMBEDDING_MATRIX.parameters(), lr = LEARNING_RATE)
    print('       [ EMBEDDING WEIGHTS INITIALIZED IN {} SECS ]'.format(round(time()-start_time, 3)))

def TrainModel ( ) :
    global INPUT_EMBEDDING_MATRIX, OUTPUT_EMBEDDING_MATRIX, OPTIMIZER_INPUT, OPTIMIZER_OUTPUT, LOSS_PROGRESS
    LOSS_PROGRESS = list()
    print_every = 500
    for epoch in range(TOTAL_EPOCHS) :
        print('\n    EPOCH {} STARTED'.format(epoch+1))
        start_time = time()
        print('       [ CONSTRUCTING MINI BATCHES ... ]')
        batches = GetBatches(BATCH_SIZE)
        print('          [ MINI BATCHES CONSTRUCTED IN {} SECS ]'.format(round(time()-start_time, 3)))
        mini_batches_seen = 0
        for input_words , target_words in batches :
            mini_batches_seen += 1
            inputs = torch.LongTensor(input_words).to(DEVICE)
            targets = torch.LongTensor(target_words).to(DEVICE)
            
            input_tensor = INPUT_EMBEDDING_MATRIX(inputs)
            output_tensor = OUTPUT_EMBEDDING_MATRIX(targets)
            noise_tensor = GetForwardNoise(inputs.shape[0], 10)
            
            loss = GetForwardLoss(input_tensor, output_tensor, noise_tensor)
            loss_value = round(loss.item(), 5)

            LOSS_PROGRESS.append(loss_value)
            if ( ( mini_batches_seen - 1 ) % print_every == 0 ) :
                print('       < {} / {} LOSS\t:\t{} >'.format(epoch+1, mini_batches_seen, loss_value))
            
            loss.backward()
            OPTIMIZER_INPUT.step()
            OPTIMIZER_INPUT.zero_grad()
            OPTIMIZER_OUTPUT.step()
            OPTIMIZER_OUTPUT.zero_grad()

def SaveModel ( model_id ) :
    out_embed_np = np.array(OUTPUT_EMBEDDING_MATRIX.weight.data.cpu())
    in_embed_np = np.array(INPUT_EMBEDDING_MATRIX.weight.data.cpu())
    word_embed = ( out_embed_np + in_embed_np ) / 2
    word2vec_model = { 'WORD_TO_ID' : WORD_TO_ID , 'ID_TO_WORD' : ID_TO_WORD , 'WORD_EMBEDDINGS' : word_embed }
    filename = 'WORD_2_VEC_MODEL_' + str(model_id).upper()
    with open(filename, 'wb') as file :
        pickle.dump(word2vec_model, file)

def TrainAndSaveWord2VecModel ( path_corpus , model_id ) :
    LoadTrainCorpus(path_corpus)
    ProcessTrainCorpus()
    CreateLookupTables()
    InitializeNoiseDistribution()
    InitializeWeightsAndOptimizers()
    TrainModel()
    SaveModel(model_id)

if __name__ == '__main__' :
    TrainAndSaveWord2VecModel("CORPUS.txt", 1)

