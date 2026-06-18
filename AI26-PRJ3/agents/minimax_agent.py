"""
MinimaxAgent
------------
عاملی مبتنی بر الگوریتم Minimax برای بازی اتلو.

نمایش بازیکنان:
    BLACK = 1
    WHITE = -1
    حریفِ هر بازیکن: opponent = -player

این عامل درخت بازی را تا عمق مشخص تولید می‌کند، در گره‌های برگ (یا در صورت
رسیدن به عمق محدود) از تابع ارزیابی استفاده می‌کند و بهترین حرکت را برمی‌گرداند.
"""

from game.othello import BLACK, WHITE, EMPTY

# ماتریس وزن موقعیتی برای صفحه‌ی ۶×۶.
# گوشه‌ها بسیار باارزش‌اند چون پایدارند؛ خانه‌های مجاور گوشه (X و C) خطرناک‌اند.
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
    ساخت ماتریس وزن موقعیتی برای صفحه‌ای با اندازه‌ی دلخواه.
    چون محیط ارزیابی لزوماً ۶×۶ نیست، برای اندازه‌های دیگر هم وزن می‌سازیم.
    """
    if size == 6:
        return WEIGHTS_6

    w = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            # فاصله از نزدیک‌ترین لبه
            edge_r = min(r, size - 1 - r)
            edge_c = min(c, size - 1 - c)
            is_corner = (edge_r == 0 and edge_c == 0)
            is_edge = (edge_r == 0 or edge_c == 0)
            # خانه‌ی مجاور گوشه (X-square / C-square)
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
        self.nodes = 0  # شمارنده‌ی گره‌های بازدیدشده (برای آزمایش‌ها)

    # ------------------------------------------------------------------ #
    # تابع ارزیابی
    # ------------------------------------------------------------------ #
    def evaluate(self, game, player):
        """
        کیفیت وضعیت فعلی بازی را از دید «player» تخمین می‌زند.
        مقدار بزرگ‌تر یعنی وضعیت بهتر برای player.

        ترکیبی وزن‌دار از چند معیار:
          ۱) اختلاف مهره‌ها (coin parity)
          ۲) امتیاز موقعیتی (positional / corner control)
          ۳) تحرک (mobility): اختلاف تعداد حرکات قانونی
          ۴) تسلط بر گوشه‌ها (corner ownership)
        وزن‌ها بسته به مرحله‌ی بازی تنظیم می‌شوند: اوایل بازی تحرک و
        موقعیت مهم‌اند، اواخر بازی اختلاف تعداد مهره‌ها تعیین‌کننده است.
        """
        opp = -player
        size = game.size
        board = game.board

        # اگر بازی تمام شده، نتیجه‌ی قطعی را با مقدار بزرگ برگردان.
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

        # تعداد خانه‌های پرشده برای تشخیص مرحله‌ی بازی
        filled = sum(cell != EMPTY for row in board for cell in row)
        total = size * size
        progress = filled / total  # بین ۰ و ۱

        # ---- ۱) اختلاف تعداد مهره‌ها ----
        my_disks = sum(cell == player for row in board for cell in row)
        opp_disks = sum(cell == opp for row in board for cell in row)
        if my_disks + opp_disks != 0:
            parity = 100 * (my_disks - opp_disks) / (my_disks + opp_disks)
        else:
            parity = 0

        # ---- ۲) امتیاز موقعیتی ----
        weights = _positional_weights(size)
        positional = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == player:
                    positional += weights[r][c]
                elif board[r][c] == opp:
                    positional -= weights[r][c]

        # ---- ۳) تحرک ----
        my_moves = len(game.get_valid_moves(player))
        opp_moves = len(game.get_valid_moves(opp))
        if my_moves + opp_moves != 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)
        else:
            mobility = 0

        # ---- ۴) تسلط بر گوشه‌ها ----
        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        my_corners = sum(board[r][c] == player for r, c in corners)
        opp_corners = sum(board[r][c] == opp for r, c in corners)
        if my_corners + opp_corners != 0:
            corner = 100 * (my_corners - opp_corners) / (my_corners + opp_corners)
        else:
            corner = 0

        # ---- ترکیب وزن‌دار بسته به مرحله‌ی بازی ----
        if progress < 0.55:        # اوایل/اواسط بازی
            w_parity, w_pos, w_mob, w_corner = 5, 10, 15, 35
        else:                      # اواخر بازی
            w_parity, w_pos, w_mob, w_corner = 25, 8, 8, 35

        return (w_parity * parity +
                w_pos * positional +
                w_mob * mobility +
                w_corner * corner)

    # ------------------------------------------------------------------ #
    # الگوریتم Minimax
    # ------------------------------------------------------------------ #
    def minimax(self, game, depth, maximizing, root_player):
        """
        بازگشتی، مقدار وضعیت و بهترین حرکت متناظر را برمی‌گرداند.
        - root_player: بازیکنی که از دید او ارزیابی می‌کنیم (همیشه ثابت).
        - maximizing: True اگر نوبتِ root_player باشد، در غیر این صورت False.
        """
        self.nodes += 1

        # شرط توقف: عمق صفر یا پایان بازی
        if depth == 0 or game.game_over():
            return self.evaluate(game, root_player), None

        current_player = root_player if maximizing else -root_player
        moves = game.get_valid_moves(current_player)

        # اگر بازیکن فعلی حرکتی ندارد → پاس (نوبت به حریف می‌رسد بدون کم‌شدن عمق چندانی)
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
        # اگر به هر دلیلی حرکتی برنگشت، اولین حرکت قانونی را انتخاب کن.
        return move if move is not None else moves[0]
