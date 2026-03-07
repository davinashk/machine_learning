import sys
import pandas as pd
#import matplotlib.pyplot as plt   # make sure this works
import torch
from torch import nn


X_mean = torch.load("./model/X_mean.pt")
X_std = torch.load("./model/X_std.pt")
y_mean = torch.load("./model/y_mean.pt")
y_std = torch.load("./model/y_std.pt")

X_data= torch.tensor([
    [5,2000,1],
    [10,4000,1]
],dtype=torch.float32)

model=nn.Linear(3,1)

prediction=model((X_data-X_mean)/X_std)

print(prediction*y_std + y_mean)
