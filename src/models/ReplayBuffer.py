import torch
from typing import Tuple

# Small batch replay buffer. Not going to do any importance sampling so we keep batch size smol
class ReplayBuffer():
    def __init__(self, encodedFeatureDims: int, numActions: int, device, bufferSize = 256, gamma = 0.9):
        self.encodedObservations = torch.zeros((bufferSize, encodedFeatureDims)).to(device)
        self.actions = torch.zeros((bufferSize)).to(device)
        self.logProbs = torch.zeros((bufferSize)).to(device)
        self.values = torch.zeros((bufferSize)).to(device)
        self.advantages = torch.zeros((bufferSize)).to(device)
        self.rewards = torch.zeros((bufferSize)).to(device)

        self.encodedFeatureDims = encodedFeatureDims
        self.numActions = numActions

        self.addedCount = 0
        self.bufferSize = bufferSize
        self.gamma = gamma

    # Do we want to do the encoding part here? Might have more paralellism and be nicer to the memory but images are expensive
    def addEntry(
            self, 
            encodedObservation: torch.Tensor,
            action: torch.Tensor,
            logProb: torch.Tensor,
            value: torch.Tensor,
            nextValue: torch.Tensor,
            reward: torch.Tensor,
            isTerminal: bool
        ):
        if self.addedCount >= self.bufferSize:
            print("Error! Buffer full")
            return

        self.encodedObservations[self.addedCount] = encodedObservation
        self.actions[self.addedCount] = action
        self.logProbs[self.addedCount] = logProb
        self.values[self.addedCount] = value
        self.rewards[self.addedCount] = reward
        
        self.advantages[self.addedCount] = int(not isTerminal) * self.gamma * nextValue + reward - value
        self.addedCount += 1

    def isFilled(self):
        return self.addedCount == self.bufferSize

    def reset(self):
        print("Resetting buffer...")
        self.addedCount = 0