# Training environment things
numBots = 200
maxSteps = 1000 # Set to -1 if you want infinite number of steps

# Rewards!
massRewardCoeff = 1
killRewardCoeff = 10
consumptionRewardCoeff = 0.4
killedPenaltyCoeff = 10
deadPenalty = 50
passivePenalty = 0.1


# Loss function things
EPSClip = 0.2
entropyCoeff = 0.1
valueCoeff = 0.1

# Learning things
gamma = 0.99
numIters = 500
numEpochs = 32
batchSize = 16
replayBufferSize = 256

learningRate = 1e-3
base_lr = 1e-3
max_lr=1e-2
min_lr=1e-7
stepSize_lr = 80
cyclical_lr = False
decay_lr = True

maxGradNorm = 10
trainFeatureExtractor = False

# Model save and load paths
startFromScratch = False
modelSaveDir = "src/models/checkpoints"
modelLoadPath = "src/models/checkpoints/model_3_pp_mass.pt"

# Feature options
usePrevFrame = False
usePrevAction = False

# Agent options
numDirections = 16