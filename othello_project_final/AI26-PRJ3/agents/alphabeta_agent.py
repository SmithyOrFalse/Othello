"""
AlphaBetaAgent
--------------
عاملی معادل MinimaxAgent اما با هرس آلفا-بتا (Alpha-Beta Pruning) برای کاهش
تعداد گره‌های بررسی‌شده. خروجی این عامل باید با Minimax یکسان باشد، فقط سریع‌تر.

نمایش بازیکنان:
    BLACK = 1
    WHITE = -1
    opponent = -player
"""

from game.othello import BLACK, WHITE, EMPTY

# همان ماتریس وزن موقعیتی برای صفحه‌ی ۶×۶
WEIGHTS_6 = [
    [100, -20, 10, 10, -20, 100],
    [-20, -50, -2, -2, -50, -20],
    [10,  -2,  -1, -1, -2,  10],
    [10,  -2,  -1, -1, -2,  10],
    [-20, -50, -2, -2, -50, -20],
    [100, -20, 10, 10, -20, 100],
]


def _positional_weights(size):
    if size == 6:
        return WEIGHTS_6
    w = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            edge_r = min(r, size - 1 - r)
            edge_c = min(c, size - 1 - c)
            is_corner = (edge_r == 0 and edge_c == 0)
            is_edge = (edge_r == 0 or edge_c == 0)
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


class AlphaBetaAgent:
    def __init__(self, depth=4):
        self.depth = depth
        self.nodes = 0  # شمارنده‌ی گره‌های بازدیدشده

    # ------------------------------------------------------------------ #
    # تابع ارزیابی (دقیقاً مشابه MinimaxAgent تا رفتار معادل باشد)
    # ------------------------------------------------------------------ #
    def evaluate(self, game, player):
        opp = -player
        size = game.size
        board = game.board

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

        filled = sum(cell != EMPTY for row in board for cell in row)
        total = size * size
        progress = filled / total

        my_disks = sum(cell == player for row in board for cell in row)
        opp_disks = sum(cell == opp for row in board for cell in row)
        parity = (100 * (my_disks - opp_disks) / (my_disks + opp_disks)
                  if my_disks + opp_disks else 0)

        weights = _positional_weights(size)
        positional = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == player:
                    positional += weights[r][c]
                elif board[r][c] == opp:
                    positional -= weights[r][c]

        my_moves = len(game.get_valid_moves(player))
        opp_moves = len(game.get_valid_moves(opp))
        mobility = (100 * (my_moves - opp_moves) / (my_moves + opp_moves)
                    if my_moves + opp_moves else 0)

        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        my_corners = sum(board[r][c] == player for r, c in corners)
        opp_corners = sum(board[r][c] == opp for r, c in corners)
        corner = (100 * (my_corners - opp_corners) / (my_corners + opp_corners)
                  if my_corners + opp_corners else 0)

        if progress < 0.55:
            w_parity, w_pos, w_mob, w_corner = 5, 10, 15, 35
        else:
            w_parity, w_pos, w_mob, w_corner = 25, 8, 8, 35

        return (w_parity * parity +
                w_pos * positional +
                w_mob * mobility +
                w_corner * corner)

    # ------------------------------------------------------------------ #
    # الگوریتم Minimax با هرس آلفا-بتا
    # ------------------------------------------------------------------ #
    def alphabeta(self, game, depth, alpha, beta, maximizing, root_player):
        """
        alpha: بهترین مقدار تضمین‌شده برای بازیکن بیشینه‌ساز تا کنون.
        beta : بهترین مقدار تضمین‌شده برای بازیکن کمینه‌ساز تا کنون.
        هرگاه beta <= alpha شود، شاخه‌های باقی‌مانده هرس می‌شوند.
        """
        self.nodes += 1

        if depth == 0 or game.game_over():
            return self.evaluate(game, root_player), None

        current_player = root_player if maximizing else -root_player
        moves = game.get_valid_moves(current_player)

        # پاس کردن در صورت نبودِ حرکت قانونی
        if not moves:
            value, _ = self.alphabeta(game, depth - 1, alpha, beta,
                                      not maximizing, root_player)
            return value, None

        best_move = None
        if maximizing:
            best_value = float('-inf')
            for move in moves:
                child = game.copy()
                child.make_move(current_player, *move)
                value, _ = self.alphabeta(child, depth - 1, alpha, beta,
                                          False, root_player)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # هرس بتا
            return best_value, best_move
        else:
            best_value = float('inf')
            for move in moves:
                child = game.copy()
                child.make_move(current_player, *move)
                value, _ = self.alphabeta(child, depth - 1, alpha, beta,
                                          True, root_player)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # هرس آلفا
            return best_value, best_move

    def choose_move(self, game, player):
        moves = game.get_valid_moves(player)
        if not moves:
            return None
        self.nodes = 0
        value, move = self.alphabeta(
            game, self.depth, float('-inf'), float('inf'), True, player
        )
        return move if move is not None else moves[0]
