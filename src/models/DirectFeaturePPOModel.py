import __future__ 

import numpy as np
from modules import *
import torch


class DirectFeaturePPOModel(torch.nn.Module):

    def __init__(
        self, 
        numDirections: int = 16,
        usePrevFrame: bool = True, 
        usePrevAction: bool = True
    ):
        super().__init__()
        self.numActions = numDirections + 2 # number of movement directions + split + shoot
        self.numEnemiesInFeature = 20
        self.numFoodInFeature = 128
        self.scalarAgentFeatureScaleFactor = 5
        self.prevActionFeatureScaleFactor = 10

        
        self.singleFrameFeatureDim = self.numEnemiesInFeature * 3 + self.numFoodInFeature * 3 + 2 * self.scalarAgentFeatureScaleFactor
        self.featureDims = self.singleFrameFeatureDim + int(usePrevFrame) * self.singleFrameFeatureDim + \
            int(usePrevAction) * self.prevActionFeatureScaleFactor
        
        self.usePrevFrame = usePrevFrame
        self.usePrevAction = usePrevAction

        self.initActor()
        self.initCritic()

    def getScaledScalarFeatureEncoding(self, scalar: float, scaleFactor: int):
        return torch.tensor([scalar for _ in range(scaleFactor)], dtype=torch.float32)

    def getSingleFrameEncoding(self, obs: dict, cooldown: float, device):
        
        # Populate player and enemy cell features
        cooldownEncodings = self.getScaledScalarFeatureEncoding(cooldown, self.scalarAgentFeatureScaleFactor)
        playerSizeEncodings = None

        enemyCellFeatures = torch.zeros(self.numEnemiesInFeature * 3)
        numAdded = 0

        for playerCellFeature in obs['player']:
            # If this cell feature is the agent's
            if playerCellFeature[3] == 1:
                playerSizeEncodings = self.getScaledScalarFeatureEncoding(playerCellFeature[2], self.scalarAgentFeatureScaleFactor)
            elif numAdded < self.numEnemiesInFeature:
                startIdx = numAdded * 3
                endIdx = startIdx + 3
                enemyCellFeatures[startIdx: endIdx] = torch.from_numpy(playerCellFeature[:3])

        # Populate food features
        foodCellFeatures = torch.zeros(self.numFoodInFeature, 3)
        if obs['food'] is not None:
            foodFeaturesToUse = len(obs['food'])
            if foodFeaturesToUse > self.numFoodInFeature:
                foodFeaturesToUse = self.numFoodInFeature
            
            foodCellFeatures[: foodFeaturesToUse] = torch.from_numpy(obs['food'][: foodFeaturesToUse])
        foodCellFeatures = torch.flatten(foodCellFeatures)

        singleFrameEncodings = torch.cat((cooldownEncodings, playerSizeEncodings, enemyCellFeatures, foodCellFeatures), 0).to(device)
        return singleFrameEncodings

    def getFullStateEncoding(self, prevFrameEncoding: torch.Tensor, currFrameEncoding: torch.Tensor, prevAction: torch.Tensor):
        if not self.usePrevAction and not self.usePrevFrame:
            return currFrameEncoding
        elif self.usePrevAction and not self.usePrevFrame:
            return torch.cat((currFrameEncoding, prevAction.repeat(self.prevActionFeatureScaleFactor)), 0)
        elif not self.usePrevAction and self.usePrevFrame:
            return torch.cat((prevFrameEncoding, currFrameEncoding), 0)
        else:
            return torch.cat((prevFrameEncoding, currFrameEncoding, prevAction.repeat(self.prevActionFeatureScaleFactor)), 0)

    # Initializes the policy
    def initActor(self):
        # Use a fully connected layer to get the action logits
        linear1 = torch.nn.Linear(self.featureDims, 128)
        torch.nn.init.kaiming_uniform_(linear1.weight, nonlinearity = 'relu')
        linear2 = torch.nn.Linear(128, 64)
        torch.nn.init.kaiming_uniform_(linear2.weight, nonlinearity = 'relu')
        linear3 = torch.nn.Linear(64, self.numActions)
        self.actor = torch.nn.Sequential(
            linear1,
            torch.nn.ReLU(),
            linear2,
            torch.nn.ReLU(),
            linear3
        )
    
    def getAction(self, statePostEncoding):
        logits = self.actor(statePostEncoding)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def getLogProbGivenAction(self, statePostEncoding, action: torch.Tensor):
        logits = self.actor(statePostEncoding)
        probs = torch.distributions.Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy()

    # Initializes the critic
    def initCritic(self):
        # Use a fully connected layer for state evaluation
        linear1 = torch.nn.Linear(self.featureDims, 64)
        torch.nn.init.kaiming_uniform_(linear1.weight, nonlinearity = 'relu')
        linear2 = torch.nn.Linear(64, 1)
        gain = torch.nn.init.calculate_gain(nonlinearity = 'linear')
        torch.nn.init.xavier_uniform_(linear2.weight, gain = gain)
        self.critic = torch.nn.Sequential(
            linear1,
            torch.nn.ReLU(),
            linear2
        )

    def getValue(self, statePostEncoding):
        return self.critic(statePostEncoding)

    def saveToFile(self, filePath):
        torch.save(self, filePath)

    def loadFromFile(self, filePath):
        self.load_state_dict(torch.load(filePath))