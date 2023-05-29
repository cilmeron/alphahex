from alphahex import Hex
import torch
from model import ResNet as HexModel
import numpy as np

def machine (board, action_set):
    game = Hex()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexModel(game, 49, 64, device)
    model.load_state_dict(torch.load('model_9.pt', map_location=device))
    state = np.array(board, dtype=float)
    neutral_state = game.change_perspective(state, -1)
    encoded_state = game.get_encoded_state(neutral_state)
    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
    policy, _ = model(tensor_state)
    valid_moves = game.get_valid_moves(neutral_state)
    probs = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
    probs *= valid_moves
    action = np.argmax(probs)
    x = action // 7;
    y = action % 7;
    tuple_xy = (x, y)
    return tuple_xy




