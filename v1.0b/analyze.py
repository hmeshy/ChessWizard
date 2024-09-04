#analyze.py
#given a pgn and model, analyze a full game with win probabs and display a graph
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import chess.pgn
#load neural network
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
    
model = torch.load('rapid2.pth') #change this to the path of the model used
model.eval()
pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/testGame.pgn") #change this to the path of the game used
game = chess.pgn.read_game(pgn)
moves = []
evalArr = []
whiteElo = int(game.headers["WhiteElo"])
blackElo = int(game.headers["BlackElo"])
scaledWhiteElo = (whiteElo - 600) / 2900
scaledBlackElo = (blackElo - 600) / 2900
whiteClock = 900
blackClock = 900
ply = 0
def evalScale(x):
    x = (2 / (math.e ** (-0.53 * x) + 1)) - 1
    return x
while game.next().next() is not None: #iterate through whole game, creating a list of attributes for each game
    game = game.next()
    ply += 1
    if ply % 2 == 0: #odd so black to move
        blackClock = game.clock()
    else:
        whiteClock = game.clock()
    scaledWhiteClock = whiteClock / 900
    scaledBlackClock = blackClock / 900 
    scaledPly = min(1, (ply-1) / 200)
    povScore = game.eval()
    if povScore.is_mate():
        score = povScore.white().mate()
        if score < 0:
            scaledEval = min(-0.99, -((score+1)/1000 + 1))
            evalArr.append(-10)
        else:
            scaledEval = max(0.99, 1 - ((score-1)/1000))
            evalArr.append(10)
    else:
        score = povScore.white().score()/100
        if score < 0:
            scaledEval = max(-0.99, evalScale(score))
            evalArr.append(max(-10,score))
        else:
            scaledEval = min(0.99, evalScale(score))
            evalArr.append(min(10,score))
    moves.append([[scaledEval, scaledWhiteElo, scaledBlackElo, scaledWhiteClock, scaledBlackClock, scaledPly]])
probab_arr = []
for move in moves: #iterate moves into the neural network, producing the win probabilities of each side.
    with torch.no_grad():
        X = torch.tensor(move)
        X = X.to(torch.double)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)
    probab_arr.append(pred_probab)
white_arr = []
draw_arr = []
for each in probab_arr:
    white_arr.append(each[0][0].item())
    draw_arr.append(each[0][1].item())
#graph the data in matplot lib
x = range(1,len(white_arr)+1)
f1 = white_arr
f2 = []
for n in range(0, len(f1)):
    f2.append(f1[n]+draw_arr[n])
# Create the plot
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot the lines
ax1.plot(x, f1, label='White/Draw', color='red')
ax1.plot(x, f2, label='Draw/Black', color='blue')

# Fill the regions
ax1.fill_between(x, 0, f1, color='white', alpha=0.5, label='White%')
ax1.fill_between(x, f1, f2, color='grey', alpha=0.5, label='Draw%')
ax1.fill_between(x, f2, 1, color='black', alpha=0.5, label='Black%')

# Adding labels and title
ax1.set_xlabel('Ply (HalfMoves)')
ax1.set_ylabel('WinProb (%)')
ax1.set_title('WinProbGraph')
ax1.legend(loc='upper left')
ax1.grid(True)


# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot the secondary graph on the secondary y-axis
ax2.plot(x, evalArr, label='SF Evaluation', color='black', linestyle='--')

# Set the secondary y-axis limits and labels
ax2.set_ylim(-10, 10)
ax2.set_ylabel('Eval')

# Adding the secondary y-axis legend
ax2.legend(loc='lower left')

# Show the plot
plt.show()
