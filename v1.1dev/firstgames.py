#get first n games
import chess.pgn
print("opening")
pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/rapideval.2.pgn")
print("done")
games = []
new_pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/test.pgn", "w", encoding="utf-8")
exporter = chess.pgn.FileExporter(new_pgn)
while True:
    game = chess.pgn.read_game(pgn)
    games.append(game)
    game.accept(exporter)
    if len(games) == 1000:
        break
