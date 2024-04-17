# Training environment things
numBots = 0
maxSteps = 1000 # Set to -1 if you want infinite number of steps

# Rewards!
massRewardCoeff = 0
killRewardCoeff = 10
consumptionRewardCoeff = 0.5
killedPenaltyCoeff = 0 # I j want a flat penalty when dying
deadPenalty = 100
passivePenalty = 0

# Loss function things
EPSClip = 0.2
entropyCoeff = 0.01
valueCoeff = 0.1

# Learning things
gamma = 0.99
numIters = 1000
numEpochs = 16
batchSize = 16
replayBufferSize = 128

learningRate = 1e-3
base_lr = 1e-3
max_lr = 1e-2
min_lr=1e-7
stepSize_lr = 80
cyclical_lr = False
decay_lr = True

maxGradNorm = 10
trainFeatureExtractor = False

# Model save and load paths
startFromScratch = True
modelSaveDir = "src/models/checkpoints"
modelLoadPath = "src/models/checkpoints/latest.pt"

# Feature options
usePrevFrame = False
usePrevAction = False

# Agent options
numDirections = 16