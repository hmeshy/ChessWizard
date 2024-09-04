#output.py -- loads trained model and gets outputs
import torch
from torch import nn
import math
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout for regularization
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = torch.load('rapid2.pth')
model.eval()
whiteElo = 1974
blackElo = 2155
scaledWhiteElo = (whiteElo - 600) / 2900
scaledBlackElo = (blackElo - 600) / 2900
ply = 0
def evalScale(x):
    x = (2 / (math.e ** (-0.53 * x) + 1)) - 1
    return x
while True:
    ply = int(input("Ply: "))
    whiteClock = float(input("WhiteClock: "))
    blackClock = float(input("BlackClock: "))
    scaledWhiteClock = whiteClock / 1800
    scaledBlackClock = blackClock / 1800 
    scaledEval = evalScale(float(input("Eval: ")))
    scaledPly = max(min(1, (ply-1) / 200),0)
    with torch.no_grad():
        X = torch.tensor([[scaledEval, scaledWhiteElo, scaledBlackElo, scaledWhiteClock, scaledBlackClock, scaledPly]])
        X = X.to(torch.double)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)
    print(pred_probab)
