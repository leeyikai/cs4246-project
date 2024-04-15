import __future__ 

from typing import Tuple
import numpy as np
from modules import *
import random
import torch
import torchvision.transforms as transforms
import torchvision.models as models


class PPOModel(torch.nn.Module):

    def __init__(self, numDirections = 16):
        super().__init__()
        self.numActions = numDirections + 2 # number of movement directions + split + shoot
        self.imageEncoderOutputDims = 1280
        self.cooldownFeatureScaleFactor = 10
        self.prevActionFeatureScaleFactor = 10
        self.singleFrameFeatureDim = self.imageEncoderOutputDims + self.cooldownFeatureScaleFactor
        self.featureDims = self.singleFrameFeatureDim * 2 + self.prevActionFeatureScaleFactor
        
        self.initImagePreprocessor()
        self.initImageEncoder()
        self.initActor()
        self.initCritic()

       
    # Returns transforms that convert the gameview to a tensor ready to be processed
    # Note that if we want to use a diff efficientNet, we need to change the Resize and CenterCrop
    # Operations accordingly: https://github.com/pytorch/vision/blob/d2bfd639e46e1c5dc3c177f889dc7750c8d137c7/references/classification/train.py#L92-L93
    def initImagePreprocessor(self):
        self.imagePreprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def getcooldownEncoding(self, cooldown: float):
        scaledCooldownArr = np.array([cooldown for _ in range(self.cooldownFeatureScaleFactor)])
        return torch.from_numpy(scaledCooldownArr)

    def initImageEncoder(self):
        # Using B0 as its the smallest and still performs better than Resnet
        efficientNet = models.efficientnet_b0(weights= models.EfficientNet_B0_Weights.DEFAULT)
        imageEncoderLayers = list(efficientNet.features.children())
        imageEncoderLayers.append(torch.nn.AdaptiveAvgPool2d(1))

        for layer in imageEncoderLayers:
            for layerParam in layer.parameters():
                layerParam.requires_grad = False
        
        self.imageEncoder = torch.nn.Sequential(*imageEncoderLayers)

    def getSingleFrameEncoding(self, view: np.ndarray, cooldown: float, device):
        preprocessedImage = self.imagePreprocessor(view.copy()).to(device)
        imageEncodings = self.imageEncoder(torch.unsqueeze(preprocessedImage, 0))
        imageEncodingsFlattened = torch.squeeze(imageEncodings)
        cooldownEncodings = self.getcooldownEncoding(cooldown).to(device)
        return torch.cat((imageEncodingsFlattened, cooldownEncodings), 0)

    def getFullStateEncoding(self, prevFrameEncoding: torch.Tensor, currFrameEncoding: torch.Tensor, prevAction: torch.Tensor):
        return torch.cat((prevFrameEncoding, currFrameEncoding, prevAction.repeat(10)), 0)

    # Initializes the policy. Uses 
    def initActor(self):
        # Use a fully connected layer to get the action logits
        linear1 = torch.nn.Linear(self.featureDims, 64)
        torch.nn.init.kaiming_uniform_(linear1.weight, nonlinearity = 'relu')
        linear2 = torch.nn.Linear(64, self.numActions)
        torch.nn.init.kaiming_uniform_(linear2.weight, nonlinearity = 'relu')
        self.actor = torch.nn.Sequential(
            linear1,
            torch.nn.ReLU(),
            linear2,
            torch.nn.ReLU()
        )
    
    def getAction(self, statePostEncoding):
        logits = self.actor(statePostEncoding)
        probs = torch.distributions.Categorical(logits=logits)
        # print("LOGITS")
        # print(logits.cpu().numpy())
        # print("PROBS")
        # print(probs.probs.cpu().numpy())
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

    @classmethod
    def fromFile(cls, filePath):
        model = PPOModel()
        model.load_state_dict(torch.load(filePath))
        return model