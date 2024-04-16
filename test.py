import torch
import numpy as np
enemyCellFeatures = torch.zeros(3 * 7)
enemyCellFeatures[0:3] = torch.from_numpy(np.array([1, 2, 3]))
print(bool(None) * enemyCellFeatures)