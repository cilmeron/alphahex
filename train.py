from alphahex import Hex
import torch
from model import ResNet as HexModel
from parallell import AlphaZero
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = {
        'C': 2,
        'num_searches': 6000,
        'num_processes': 10,
        'num_iterations':5,
        'num_selfPlay_iterations': 4,
        'num_epochs':10,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    game = Hex()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexModel(game, 49, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    alphaZero = AlphaZero(model, optimizer, game, args)

    alphaZero.learn()