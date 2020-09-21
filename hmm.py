from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for s in range(0,S):
            alpha[s,0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]
        
        for l in range(1,L):
            for s in range(0,S):
                alpha[s,l] = self.B[s, self.obs_dict[Osequence[l]]] * np.sum(self.A[:,s] * alpha[:,l-1])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        beta[:,L-1] = 1
        
        for l in range(L-2,-1,-1):
            for s in range(0,S):
                beta[s,l] = np.sum(self.A[s,:] * self.B[:,self.obs_dict[Osequence[l+1]]] * beta[:,l+1])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        prob = np.sum(self.forward(Osequence)[:,-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sequence_prob = self.sequence_prob(Osequence)

        prob = alpha * beta / sequence_prob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        P = self.sequence_prob(Osequence)
        
        for l in range(0,L-1):
            for s1 in range(0,S):
                for s2 in range(0,S):
                    prob[s1,s2,l] = alpha[s1,l]*self.A[s1,s2]*self.B[s2,self.obs_dict[Osequence[l+1]]]*beta[s2,l+1]/P
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        temp = np.zeros([S, L-1])
        
        for s in range(0,S):
            delta[s, 0] = self.pi[s] * self.B[s, self.obs_dict[Osequence[0]]]
        
        for l in range(1,L):
            for s in range(0,S):
                product = self.A[:,s] * delta[:,l-1]
                delta[s,l] = self.B[s,self.obs_dict[Osequence[l]]] * np.max(product)
                temp[s,l-1] = np.argmax(product)
        
        path.append(np.argmax(delta[:,L-1]))
        for l in range(L-2,-1,-1):
            path.append(int(temp[path[-1],l]))
        
        j = 0
        for i in path:
            for key, value in self.state_dict.items():
                if value == i:
                    path[j] = key
            j = j+1

        path = path[::-1]
        ###################################################
        return path
