import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import copy
from numpy.linalg import norm

# 损失函数
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# Define memory class.
class Memory(object):
    def __init__(self, size):
        # Initially empty.
        self.memory = {}
        # The number of (key,logits) pairs stored in memory.
        self.size = size
        
    def get_size(self):
        # Return the number of (key,logits) pairs in memory.
        return len(self.memory) 
    
    def push(self, keys, logits): # Push a (key,logits) pair into memory.
        dimension =  16*3000
        keys=keys.reshape(len(keys)*35,dimension)
        for i, key in enumerate(keys):
           
            if len(self.memory.keys())>self.size:
                self.memory.pop(list(self.memory)[0]) 
            # Update (key,logits) pairs in the memory bank according to the First In First Out (FIFO) principle.
            self.memory.update(
                {key.reshape(dimension).tobytes(): (logits[i])})

    # Calculate the weighted average of the prediction masks of all neighbors corresponding to a key.
    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]
        ensemble_prediction = ensemble_prediction.cpu()
        return torch.FloatTensor(ensemble_prediction)
    

    def get_neighbours(self, keys, k):
        # Return the weighted average output of the k nearest neighbor samples of each of all current samples.
        # Here you get a neighbor for each key, and each key in the same batch corresponds to the same memory bank.
        samples = []
        dimension =  16*3000
        keys = keys.reshape(len(keys)*35, dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, dimension)
        for key in keys:
          
            similarity_scores = np.dot(self.all_keys, key.T)/(norm(self.all_keys, axis=1) * norm(key.T) )
            # Get the keys that are most similar to the current sample.
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            # Put the values corresponding to the keys of these neighbours into the list.
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
        
            attention_weight = np.dot(K_neighbour_keys, key.T) /(norm(K_neighbour_keys, axis=1) * norm(key.T) )
            if len(neighbours)==0:
                return samples
            batch = self._prepare_batch(neighbours,attention_weight)
            samples.append(batch)

        return torch.stack(samples)
