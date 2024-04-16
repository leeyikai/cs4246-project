# Training environment things
numBots = 10
maxSteps = 1000 # Set to -1 if you want infinite number of steps

# Rewards!
massRewardCoeff = 0
killRewardCoeff = 10
consumptionRewardCoeff = 0.1
killedPenaltyCoeff = 10
deadPenalty = 50
passivePenalty = 0

# Loss function things
EPSClip = 0.2
entropyCoeff = 0.01
valueCoeff = 0.1

# Learning things
gamma = 0.99
numIters = 1000
numEpochs = 32
batchSize = 64
replayBufferSize = 256
learningRate = 1e-3
maxGradNorm = 10
trainFeatureExtractor = False

# Model save and load paths
startFromScratch = True
modelSaveDir = "src/models/checkpoints"
modelLoadPath = "src/models/checkpoints/Direct.pt"

# Feature options
usePrevFrame = False
usePrevAction = True

# Agent options
numDirections = 16