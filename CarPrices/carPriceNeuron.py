import pandas as pd
import torch
from torch import nn
#import matplotlib.pyplot as plt   # make sure this works

df = pd.read_csv("./data/used_cars.csv")

age=df["model_year"].max() - df["model_year"]

milage=df["milage"]
milage=milage.str.replace(",","").str.replace(" mi.","").astype(int)
price=df["price"]
price=price.str.replace(",","").str.replace("$","").astype(int)

X=torch.column_stack([
    torch.tensor(age,dtype=torch.float32),
    torch.tensor(milage,dtype=torch.float32)
    ])

y=torch.tensor(price,dtype=torch.float32)\
    .reshape(-1,1)

model=nn.Linear(2,1)
loss_func=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.0000000001)

for epoch in range(1000):
    y_pred=model(X)
    loss=loss_func(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #if epoch % 100 == 0:
     #   print(f"Epoch {epoch}, Loss: {loss.item()}")
      #  print(model.weight.item(), model.bias.item())


prediction=model(torch.tensor([
    [5,2000]
],dtype=torch.float32))

print(prediction)






