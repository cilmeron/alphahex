from alphahex import Hex
import torch
from model import ResNet as HexModel
import numpy as np
from parallell import MCTS

if __name__ == '__main__':
    game = Hex()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexModel(game, 49, 64, device)
    model.load_state_dict(torch.load('model_0.pt', map_location=device))
    model.eval()
    state = game.get_init_state()
    player=1
    game.draw(player, state)
    while True:
        if player == 1:
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
            print(state)
            neutral_state = game.change_perspective(state, player)
            encoded_state = game.get_encoded_state(neutral_state)
            tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
            policy, value = model(tensor_state)
            valid_moves = game.get_valid_moves(neutral_state)
            print(valid_moves)
            value = value.item()
            probs = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
            probs *= valid_moves
            action = np.argmax(probs)

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



