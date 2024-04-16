import torch
from typing import Tuple
import numpy as np

# Small batch replay buffer. Not going to do any importance sampling so we keep batch size smol
class ReplayBuffer():
    def __init__(self, encodedFeatureDims: int, numActions: int, device, bufferSize = 256, gamma = 0.9):
        self.encodedObservations = torch.zeros((bufferSize, encodedFeatureDims)).to(device)
        self.actions = torch.zeros((bufferSize)).to(device)
        self.prevActions = torch.zeros((bufferSize)).to(device)
        self.logProbs = torch.zeros((bufferSize)).to(device)
        self.values = torch.zeros((bufferSize)).to(device)
        self.advantages = torch.zeros((bufferSize)).to(device)
        self.rewards = torch.zeros((bufferSize)).to(device)
        self.ejectCooldowns = np.empty(bufferSize, dtype=int)
        self.prevEjectCooldowns = np.empty(bufferSize, dtype=int)
        self.prevViews = np.empty(bufferSize, dtype=np.ndarray)
        self.views = np.empty(bufferSize, dtype=np.ndarray)

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
            isTerminal: bool,
            ejectCooldown: int = None,
            prevEjectCooldown: int = None,
            prevAction: torch.Tensor = None,
            prevView: np.ndarray = None,
            view: np.ndarray = None
        ):
        if self.addedCount >= self.bufferSize:
            print("Error! Buffer full")
            return

        if prevAction is not None:
            self.prevActions[self.addedCount] = prevAction

        if encodedObservation is not None:
            self.encodedObservations[self.addedCount] = encodedObservation

        self.actions[self.addedCount] = action
        self.logProbs[self.addedCount] = logProb
        self.values[self.addedCount] = value
        self.rewards[self.addedCount] = reward
        
        if nextValue is not None:
            self.advantages[self.addedCount] = int(not isTerminal) * self.gamma * nextValue + reward - value
        else:
            self.advantages[self.addedCount] = reward - value

        if ejectCooldown is not None:
            self.ejectCooldowns[self.addedCount] = ejectCooldown
        
        if prevEjectCooldown is not None:
            self.prevEjectCooldowns[self.addedCount] = prevEjectCooldown

        if view is not None:
            self.views[self.addedCount] = view

        if prevView is not None:
            self.prevViews[self.addedCount] = prevView
            
        self.addedCount += 1

    def isFilled(self):
        return self.addedCount == self.bufferSize

    def reset(self):
        print("Resetting buffer...")
        self.addedCount = 0