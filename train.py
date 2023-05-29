from alphahex import Hex
from model import ResNet as HexModel
from parallell import AlphaZero
import torch
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = {
        'C': 2,
        'num_searches': 1500,
        'num_processes': 5,
        'num_iterations': 10,
        'num_selfPlay_iterations': 40,
        'num_epochs': 10,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    game = Hex()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexModel(game, 49, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    last_save = 9
    models = torch.load(f"model_{last_save}.pt")
    opts = torch.load(f"optimizer_{last_save}.pt")
    model.load_state_dict(models)
    optimizer.load_state_dict(opts)
    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()