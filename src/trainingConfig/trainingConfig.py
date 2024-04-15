# Training environment things
numBots = 0

# Rewards!
massRewardCoeff = 0
killRewardCoeff = 10
consumptionRewardCoeff = 0.1
killedPenaltyCoeff = 10
deadPenalty = 500
passivePenalty = 0.001

# Loss function things
EPSClip = 0.2
entropyCoeff = 0.01
valueCoeff = 0.08

# Learning things
gamma = 0.99
numIters = 1000
numEpochs = 16
batchSize = 32
replayBufferSize = 256
learningRate = 1e-3
maxGradNorm = 10

# Model save and load paths
startFromScratch = True
modelSaveDir = "src/models/checkpoints"
modelLoadDir = "src/models/checkpoints/latest.pt"