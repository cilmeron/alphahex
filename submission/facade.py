import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#stupid example
def agent (board, action_set):
	game = Hex()
	if np.sum(board)%2 == 0:
		player = 1
	else:
		player = -1
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = HexModel(game, 49, 64, device)
	model.load_state_dict(torch.load('fhtw_hex/submission/model_9.pt', map_location=device))
	state = np.array(board, dtype=float)
	neutral_state = game.change_perspective(state, player)
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

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.


import numpy as np


class Hex:
	def __init__(self):
		self.row_count = 7
		self.column_count = 7
		self.action_size = self.row_count * self.column_count

	def __repr__(self):
		return "Hex"

	def get_init_state(self):
		return np.zeros((self.row_count, self.column_count))

	def get_initial_state(self):
		return self.get_init_state()

	def get_next_state(self, state, action, player):
		row = action // self.column_count
		column = action % self.column_count
		state[row, column] = player
		return state

	@staticmethod
	def get_valid_moves(state):
		return (state.reshape(-1) == 0).astype(np.uint8)

	def get_neighbors(self, row, col):
		neighbors = []
		size = self.row_count  # Assuming the Hex board is a square shape
		# Check the six neighboring positions
		directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
		for direction in directions:
			dx, dy = direction
			new_row = row + dx
			new_col = col + dy

			# Check if the new position is within the board boundaries
			if 0 <= new_row < size and 0 <= new_col < size:
				neighbors.append((new_row, new_col))

		return neighbors

	def find_path(self, state, player):
		visited = set()
		queue = []
		size = self.row_count

		if player == 1:
			# Player 1 wants to connect the left and right sides
			for i in range(size):
				if state[i, 0] == player:
					queue.append((i, 0))
					visited.add((i, 0))
		else:
			# Player -1 wants to connect the top and bottom sides
			for j in range(size):
				if state[0, j] == player:
					queue.append((0, j))
					visited.add((0, j))

		while queue:
			row, col = queue.pop(0)

			if (player == 1 and col == size - 1) or (player == -1 and row == size - 1):
				return True

			neighbors = self.get_neighbors(row, col)
			for neighbor in neighbors:
				if neighbor not in visited and state[neighbor[0], neighbor[1]] == player:
					queue.append(neighbor)
					visited.add(neighbor)

		return False

	def draw(self, local_player, local_state):
		if local_player == 1:
			local_state = np.rot90(local_state)
			local_state = np.flipud(local_state)
		for i in range(self.row_count):
			print(chr(ord('A') + i), end=" ")
		print("\n", end="")
		for i in range(self.row_count):
			for a in range(i):
				print(" ", end="")
			for j in range(self.column_count):
				if local_state[i, j] == 1:
					print("b", end=" ")
				elif local_state[i, j] == -1:
					print("w", end=" ")
				else:
					print("_", end=" ")
			print(i + 1, "\n", end="")

	def check_win(self, state, action):
		if action is None:
			return False

		row = action // self.column_count
		col = action % self.column_count
		player = state[row, col]

		return self.find_path(state, player)

	def get_value_and_terminated(self, state, action):
		if self.check_win(state, action):
			return 1, True
		if np.sum(self.get_valid_moves(state)) == 0:
			return 0, True
		return 0, False

	@staticmethod
	def get_opponent(player):
		return -player

	@staticmethod
	def get_opponent_value(value):
		return -value

	@staticmethod
	def change_perspective(state, player):
		return state * player

	@staticmethod
	def get_state_encoded(state):
		output_state = np.stack(
			(state == -1, state == 0, state == 1)
		).astype(np.float32)
		return output_state

	def get_encoded_state(self, state):
		return self.get_state_encoded(state)


class ConnectFour:
	def __init__(self):
		self.row_count = 6
		self.column_count = 7
		self.action_size = self.column_count
		self.in_a_row = 4

	def __repr__(self):
		return "ConnectFour"

	def get_initial_state(self):
		return np.zeros((self.row_count, self.column_count))

	def get_next_state(self, state, action, player):
		row = np.max(np.where(state[:, action] == 0))
		state[row, action] = player
		return state

	def get_valid_moves(self, state):
		return (state[0] == 0).astype(np.uint8)

	def check_win(self, state, action):
		if action == None:
			return False

		row = np.min(np.where(state[:, action] != 0))
		column = action
		player = state[row][column]

		def count(offset_row, offset_column):
			for i in range(1, self.in_a_row):
				r = row + offset_row * i
				c = action + offset_column * i
				if (
						r < 0
						or r >= self.row_count
						or c < 0
						or c >= self.column_count
						or state[r][c] != player
				):
					return i - 1
			return self.in_a_row - 1

		return (
				count(1, 0) >= self.in_a_row - 1  # vertical
				or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1  # horizontal
				or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1  # top left diagonal
				or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1  # top right diagonal
		)

	def get_value_and_terminated(self, state, action):
		if self.check_win(state, action):
			return 1, True
		if np.sum(self.get_valid_moves(state)) == 0:
			return 0, True
		return 0, False

	def get_opponent(self, player):
		return -player

	def get_opponent_value(self, value):
		return -value

	def change_perspective(self, state, player):
		return state * player

	def get_encoded_state(self, state):
		encoded_state = np.stack(
			(state == -1, state == 0, state == 1)
		).astype(np.float32)

		if len(state.shape) == 3:
			encoded_state = np.swapaxes(encoded_state, 0, 1)

		return encoded_state
		
		
class HexModel(nn.Module):
	def __init__(self, game, num_resBlocks, num_hidden, device):
		super().__init__()

		self.device = device
		self.startBlock = nn.Sequential(
			nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
			nn.BatchNorm2d(num_hidden),
			nn.ReLU()
		)

		self.backBone = nn.ModuleList(
			[ResBlock(num_hidden) for i in range(num_resBlocks)]
		)

		self.policyHead = nn.Sequential(
			nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(32 * game.row_count * game.column_count, game.action_size)
		)

		self.valueHead = nn.Sequential(
			nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
			nn.BatchNorm2d(3),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3 * game.row_count * game.column_count, 1),
			nn.Tanh()
		)

		self.to(device)

	def forward(self, x):
		x = self.startBlock(x)
		for resBlock in self.backBone:
			x = resBlock(x)
		policy = self.policyHead(x)
		value = self.valueHead(x)
		return policy, value


class ResBlock(nn.Module):
	def __init__(self, num_hidden):
		super().__init__()
		self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_hidden)
		self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(num_hidden)

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x += residual
		x = F.relu(x)
		return x