#creates and saves numpy array with necessary data, then saves each line as a new file
#also must save labels file at the same time
#data to be saved:
#evalulation; always white pov, normalized using sigmoid e^-0.46; M1 1, M2 0.999, M3 0.998... M10 0.991, +10 or > 0.99
#player elo; linear 600 is 0, 3500 is 1
#time: 30 mins+ - 1800 is 1, 0 mins is 0, linearized
#ply count: 1 is 0, 201+ is one, linearized
#labels 0 = win, 1 = draw, 2 = loss
import chess.pgn
import numpy as np
import math
print("opening")
dataLength = 10000
data = [[0,0,0,0,0,0]]
labels = [["move0.csv", 0]] #sample data
pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/test.pgn")
print("done")

def evalScale(x):
    x = (2 / (math.e ** (-0.53 * x) + 1)) - 1
    return x
for gameN in range(1,1001):
    print(gameN)
    game = chess.pgn.read_game(pgn)
    correspondence = game.headers["TimeControl"] == "-"
    result = game.headers["Result"]
    if result == "1-0":
        result = 0
    elif result == "0-1":
        result = 2
    else:
        result = 1
    whiteElo = int(game.headers["WhiteElo"])
    blackElo = int(game.headers["BlackElo"])
    scaledWhiteElo = (whiteElo - 600) / 2900
    scaledBlackElo = (blackElo - 600) / 2900
    blackClock = 2
    whiteClock = 2
    scaledBlackClock = 1
    scaledWhiteClock = 1
    while game.next().next() != None:
        game = game.next()
        ply = game.ply()
        scaledPly = min(1, (ply-1) / 200)
        if not correspondence:
            if ply % 2 == 0: #odd so black to move
                blackClock = game.clock()
                if whiteClock == 2:
                    whiteClock = game.clock()
            else:
                whiteClock = game.clock()
                if blackClock == 2:
                    blackClock = game.clock()   
            scaledWhiteClock = whiteClock / 900
            scaledBlackClock = blackClock / 900    
        povScore = game.eval()
        if povScore.is_mate():
            score = povScore.white().mate()
            if score < 0:
                scaledEval = min(-0.99, -((score+1)/1000 + 1))
            else:
                scaledEval = max(0.99, 1 - ((score-1)/1000))
        else:
            score = povScore.white().score()/100
            if score < 0:
                scaledEval = max(-0.99, evalScale(score))
            else:
                scaledEval = min(0.99, evalScale(score))  
        data.append([[scaledEval, scaledWhiteElo, scaledBlackElo, scaledWhiteClock, scaledBlackClock, scaledPly]])
        info = "game" + str(gameN) + "move" + str(ply) + ".csv"
        labels.append([info, result])
for i in range(len(data)):
    np.savetxt("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/data/testData/" + labels[i][0], data[i])
np.savetxt("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/data/testLabels.csv", labels, delimiter = ", ", fmt = '% s')