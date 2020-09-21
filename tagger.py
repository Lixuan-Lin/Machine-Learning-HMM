import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    S = len(tags)

    obs_dict = {}
    l = 0
    for line in train_data:
        for word in line.words:
            if word not in obs_dict:
                obs_dict[word] = l
                l = l+1
    L = l
    
    state_dict = {}
    for i, tag in enumerate(tags):
        state_dict[tag] = i
    
    FirstTag = {}
    for line in train_data:
        if tags.index(line.tags[0]) not in FirstTag:
            FirstTag[tags.index(line.tags[0])] = 1
        else:
            FirstTag[tags.index(line.tags[0])] += 1
    
    pi = np.zeros([S])
    FirstTotal = sum(FirstTag.values())
    for i in range(S):
        pi[i] = FirstTag[i]/FirstTotal
    
    A = np.zeros([S,S])
    B = np.zeros([S,L])
    for line in train_data:
        for m in range(line.length-1):
            A[state_dict[line.tags[m]],state_dict[line.tags[m+1]]] += 1 
        for n in range(line.length):
            B[state_dict[line.tags[n]],obs_dict[line.words[n]]] += 1
    for s in range(S):
        A[s] = np.divide(A[s], np.sum(A[s]))
        B[s] = np.divide(B[s], np.sum(B[s]))
    
    model = HMM(pi, A, B, obs_dict, state_dict)
    ###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here
    S = len(model.pi)
    
    for line in test_data:
        for word in line.words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = len(model.obs_dict)
                model.B = np.column_stack((model.B, np.full([S,1],0.000001)))
        tagging.append(model.viterbi(line.words))
    ###################################################
    return tagging
