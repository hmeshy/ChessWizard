#create new file with only chess games with evaluation
import chess.pgn
print("opening")
pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/lichess_db_standard_rated_2024-07.pgn")
print("done")
games = []
new_pgn = open("C:/Users/hmesh/OneDrive/Documents/PGNs/lichessDataBase/rapid.pgn", "w", encoding="utf-8")
exporter = chess.pgn.FileExporter(new_pgn)
while True:
    offset = pgn.tell()
    game = chess.pgn.read_headers(pgn)
    if game is None:
        break
    elif "900+10" in game.get("TimeControl","?"):
        games.append(offset)
        if len(games) % 1000 == 0:
            print(len(games))
print(len(games))
for offset in games:
    pgn.seek(offset)
    game = chess.pgn.read_game(pgn)
    game.accept(exporter)
