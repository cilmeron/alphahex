from alphahex import Hex
import torch
from model import ResNet as HexModel
from mctsp import MCTS
import numpy as np

args = {
    'C': 2,
    'num_searches': 400,
    'num_iterations': 3,
    'num_selfPlay_iterations': 100,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

game = Hex()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HexModel(game, 9, 128, device)
model.load_state_dict(torch.load('model_2.pt'))
model.eval()

mcts = MCTS(game, args, model)
state = game.get_init_state()
player=-1
game.draw(player, state)
while True:
    if player == -1:
        valid_movies = game.get_valid_moves(state)
        start_letter = ord('A')
        print("valid_moves=", end="")
        for i in range(valid_movies.size):
            if valid_movies[i] == 1:
                y = i // game.column_count
                x = i % game.row_count
                print(chr(start_letter+x), end="")
                print(y+1, end=" ")
        print("\n", end="")
        action = input(f"{player}:")
        x=(ord(action[0])-ord('A'))
        y=int(action[1])-1
        print("\n", x, " ", y)
        coord = x+game.row_count*y
        action = coord
        if action>(game.row_count*game.column_count):
            print("invalid action")
            continue
        if valid_movies[action] == 0:
            print("invalid action")
            continue
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = game.get_next_state(state, action, player)

    value, is_terminal = game.get_value_and_terminated(state, action)

    if is_terminal:
        if value == 1:
            print("Player ", player, " has won the game")
            game.draw(player, state)
        else:
            print("Game ended in a draw")
            game.draw(player, state)
        break
    game.draw(player, state)
    player = game.get_opponent(player)



