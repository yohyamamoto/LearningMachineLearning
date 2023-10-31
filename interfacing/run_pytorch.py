import copy
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

#print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Assuming that you already performed the scaler transformation
dataset = np.loadtxt('torch_in.csv',delimiter=',')
X = dataset[:,0:6]


#scaler = StandardScaler()
#scaler.fit(X_raw)

#print('mean',scaler.mean_)
#print('var_',scaler.var_)
#X = scaler.transform(X_raw)[:,0:6]

#convert a tensor from numpy arrays
X = torch.tensor(X, dtype=torch.float32, device=device) #.reshape(-1,1)
#print(X)

model = nn.Sequential(
    nn.Linear(6, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
)
model = model.to(device)
#print(model)


#restore model
model.load_state_dict(torch.load('torch_nn.pt'))
model.eval()

# make prediction with the model
predictions = model(X)

np.set_printoptions(precision=16)

#for i in range(10):
#    print('%f' % (predictions[i]))

p_np = predictions.detach().numpy()
df = pd.DataFrame(p_np)
df.to_csv("torch_out.csv",index=False)


