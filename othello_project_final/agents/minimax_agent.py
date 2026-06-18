"""
MinimaxAgent
------------
A Minimax-based agent for the Othello board game.

Player representation:
    BLACK = 1
    WHITE = -1
    Opponent of any player: opponent = -player

The agent builds a game tree up to a specified depth. At leaf nodes (or when
the depth limit is reached) it uses the evaluation function, then returns the
best move found.
"""

from game.othello import BLACK, WHITE, EMPTY

# Positional weight matrix for a 6x6 board.
# Corners are very valuable (stable and permanent); X/C squares next to corners are dangerous.
WEIGHTS_6 = [
    [100, -20, 10, 10, -20, 100],
    [-20, -50, -2, -2, -50, -20],
    [10,  -2,  -1, -1, -2,  10],
    [10,  -2,  -1, -1, -2,  10],
    [-20, -50, -2, -2, -50, -20],
    [100, -20, 10, 10, -20, 100],
]


def _positional_weights(size):
    """
    Build a positional weight matrix for an arbitrary board size.
    Since the evaluation environment is not necessarily 6x6, we generate
    weights dynamically for other sizes.
    """
    if size == 6:
        return WEIGHTS_6

    w = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            # Distance from the nearest edge
            edge_r = min(r, size - 1 - r)
            edge_c = min(c, size - 1 - c)
            is_corner = (edge_r == 0 and edge_c == 0)
            is_edge = (edge_r == 0 or edge_c == 0)
            # Squares adjacent to a corner (X-square / C-square)
            adj_corner = (edge_r <= 1 and edge_c <= 1) and not is_corner

            if is_corner:
                w[r][c] = 100
            elif adj_corner:
                w[r][c] = -30
            elif is_edge:
                w[r][c] = 8
            else:
                w[r][c] = 1
    return w


class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth
        self.nodes = 0  # counter for visited nodes (used in experiments)

    # ------------------------------------------------------------------ #
    # Evaluation function
    # ------------------------------------------------------------------ #
    def evaluate(self, game, player):
        """
        Estimate the quality of the current game state from the perspective
        of 'player'. A higher value means a better position for player.

        Uses a weighted combination of four heuristics:
          1) Coin parity      - relative disc count difference
          2) Positional score - weighted sum of occupied squares
          3) Mobility         - difference in number of legal moves
          4) Corner ownership - difference in corners captured
        Weights are adjusted based on game phase: early/mid game favours
        mobility and corners; late game favours coin parity.
        """
        opp = -player
        size = game.size
        board = game.board

        # Terminal state: return a definitive large value.
        if game.game_over():
            b, w = game.score()
            my = b if player == BLACK else w
            their = w if player == BLACK else b
            if my > their:
                return 10_000 + (my - their)
            elif my < their:
                return -10_000 + (my - their)
            else:
                return 0

        # Count filled squares to determine game phase
        filled = sum(cell != EMPTY for row in board for cell in row)
        total = size * size
        progress = filled / total  # value between 0 and 1

        # ---- 1) Coin parity ----
        my_disks = sum(cell == player for row in board for cell in row)
        opp_disks = sum(cell == opp for row in board for cell in row)
        if my_disks + opp_disks != 0:
            parity = 100 * (my_disks - opp_disks) / (my_disks + opp_disks)
        else:
            parity = 0

        # ---- 2) Positional score ----
        weights = _positional_weights(size)
        positional = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == player:
                    positional += weights[r][c]
                elif board[r][c] == opp:
                    positional -= weights[r][c]

        # ---- 3) Mobility ----
        my_moves = len(game.get_valid_moves(player))
        opp_moves = len(game.get_valid_moves(opp))
        if my_moves + opp_moves != 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)
        else:
            mobility = 0

        # ---- 4) Corner ownership ----
        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        my_corners = sum(board[r][c] == player for r, c in corners)
        opp_corners = sum(board[r][c] == opp for r, c in corners)
        if my_corners + opp_corners != 0:
            corner = 100 * (my_corners - opp_corners) / (my_corners + opp_corners)
        else:
            corner = 0

        # ---- Weighted combination based on game phase ----
        if progress < 0.55:        # early / mid game
            w_parity, w_pos, w_mob, w_corner = 5, 10, 15, 35
        else:                      # late game
            w_parity, w_pos, w_mob, w_corner = 25, 8, 8, 35

        return (w_parity * parity +
                w_pos * positional +
                w_mob * mobility +
                w_corner * corner)

    # ------------------------------------------------------------------ #
    # Minimax algorithm
    # ------------------------------------------------------------------ #
    def minimax(self, game, depth, maximizing, root_player):
        """
        Recursive Minimax. Returns (value, best_move).
        - root_player: the player from whose perspective we evaluate (constant).
        - maximizing: True when it is root_player's turn, False otherwise.
        """
        self.nodes += 1

        # Base case: depth limit reached or game over
        if depth == 0 or game.game_over():
            return self.evaluate(game, root_player), None

        current_player = root_player if maximizing else -root_player
        moves = game.get_valid_moves(current_player)

        # No legal moves: pass turn to opponent (depth decremented by 1)
        if not moves:
            value, _ = self.minimax(game, depth - 1, not maximizing, root_player)
            return value, None

        best_move = None
        if maximizing:
            best_value = float('-inf')
            for move in moves:
                child = game.copy()
                child.make_move(current_player, *move)
                value, _ = self.minimax(child, depth - 1, False, root_player)
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move
        else:
            best_value = float('inf')
            for move in moves:
                child = game.copy()
                child.make_move(current_player, *move)
                value, _ = self.minimax(child, depth - 1, True, root_player)
                if value < best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move

    def choose_move(self, game, player):
        moves = game.get_valid_moves(player)
        if not moves:
            return None
        self.nodes = 0
        value, move = self.minimax(game, self.depth, True, player)
        # Fallback: if no move returned, pick the first legal move
        return move if move is not None else moves[0]
