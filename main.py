import torch
from torch import nn


x1=torch.tensor([[10.0]],dtype=torch.float32)
y1=torch.tensor([[50.0]],dtype=torch.float32)
x2=torch.tensor([[37.8]],dtype=torch.float32)
y2=torch.tensor([[100.0]],dtype=torch.float32)
  # shape (batch_size=1, in_features=1)
model=nn.Linear(1,1)

loss_func=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(100000):
    y_pred=model(x1)
    loss=loss_func(y_pred, y1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred=model(x1)
    loss=loss_func(y_pred, y1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(model.weight.item(), model.bias.item())



