import pandas as pd
#import matplotlib.pyplot as plt   # make sure this works
import torch
from torch import nn
import sys
import os


# Fix randomness for reproducible predictions
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

df = pd.read_csv("./data/used_cars.csv")

age=df["model_year"].max() - df["model_year"]

milage=df["milage"]
milage=milage.str.replace(",","").str.replace(" mi.","").astype(int)
price=df["price"]
price=price.str.replace(",","").str.replace("$","").astype(int)

accident_free = df["accident"] == "None reported"
accident_free = accident_free.astype(int)

if not os.path.isdir("./model"):
    os.mkdir("./model")


X=torch.column_stack([
    torch.tensor(age,dtype=torch.float32),
    torch.tensor(milage,dtype=torch.float32),
    torch.tensor(accident_free,dtype=torch.float32)
    ])

X_mean=X.mean(axis=0)
X_std=X.std(axis=0)
torch.save(X_mean,"./model/X_mean.pt")
torch.save(X_std,"./model/X_std.pt")

X=(X-X_mean)/X_std

y=torch.tensor(price,dtype=torch.float32)\
    .reshape(-1,1)

y_mean = y.mean()
y_std = y.std()

torch.save(y_mean,"./model/y_mean.pt")
torch.save(y_std,"./model/y_std.pt")

y = (y-y_mean)/y_std

model=nn.Linear(3,1)
loss_func=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.0001)

#losses = []

for epoch in range(2500):
    y_pred=model(X)
    loss=loss_func(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #losses.append(loss.item())
    
    #if epoch % 100 == 0:
       #print(f"Epoch {epoch}, Loss: {loss.item()}")
       #print(model.weight.item(), model.bias.item())

torch.save(model.state_dict(),"./model/model.pt")

print(torch.load("./model/model.pt"))








