# ChessWizard
neural network model to calculate the win probability throughout a chess game

Rather than just pre-game win probabilities based on elo, we thought it would be a fun project to use a neural network to generate live win probabilities based on elo, time, evaluation (only those three for our most simple model), etc...

v1 is complete, with a quite good model for 15+10 rapid chess, despite running on quite limited information Todo: Make a code to automatically analyze a game with v1 and display a graph; clicking on a ply ideally displays the clock times and eval (possibly add a board but like no need) Make v2, with more factors including: "Mobility" - # of psuedolegal moves for each side, assuming it was their turn to play. "Material" - Amount of material on the board, counted the normal way, for each color possibly - "Increment" - may just be better to include data in the same time control to simplify the model

Train v1 and v2 on WCC, Candidates, and Candidates Qualifying Tournaments (Grand Prix, World Cup, etc.) (classical only?) in time for WCC 2024 (November)

Also, try different optimizers to try and create optimal models

v1 results in training: Accuracy: 65.2%, Avg loss: 0.778032 v1 new model type Accuracy: 66.2%, Avg loss: 0.709914

Also, simply based off one game, v1b delivers much more reasonable results (although perhaps not perfect results) for the data given
