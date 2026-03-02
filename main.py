import torch
from torch import nn


x=torch.tensor([[10.0],
                [37.8]],dtype=torch.float32)
y=torch.tensor([[50.0],
                [100.0]],dtype=torch.float32)

model=nn.Linear(1,1)

loss_func=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(150000):
    y_pred=model(x)
    loss=loss_func(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(model.weight.item(), model.bias.item())



