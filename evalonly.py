#create new file with only chess games with evaluation
import chess.pgn
print("opening")
pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/rapid.pgn")
print("done")
games = []
new_pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/rapideval.pgn", "w", encoding="utf-8")
exporter = chess.pgn.FileExporter(new_pgn)
while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    elif game.next() is not None and game.next().eval() is not None:
        games.append(game)
        game.accept(exporter)
        if len(games) % 1000 == 0:
            print(len(games))
print(len(games))

