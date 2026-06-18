"""
اجرای آزمایش‌ها: نرخ برد عامل‌ها در برابر Random و Greedy،
و مقایسه‌ی تعداد گره‌ها و زمان اجرای Minimax و Alpha-Beta.
"""
import time, random
from game.othello import Othello, BLACK, WHITE
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent


def play_game(agent_black, agent_white, size=6):
    game = Othello(size)
    player = BLACK
    while not game.game_over():
        moves = game.get_valid_moves(player)
        if moves:
            agent = agent_black if player == BLACK else agent_white
            move = agent.choose_move(game, player)
            if move is not None:
                game.make_move(player, *move)
        player = WHITE if player == BLACK else BLACK
    return game.score()


def winrate(agent_factory, opponent_factory, n=20, depth_label=""):
    """ عاملِ ما به نوبت سیاه و سفید بازی می‌کند تا منصفانه باشد. """
    wins = draws = 0
    for i in range(n):
        if i % 2 == 0:
            b, w = play_game(agent_factory(), opponent_factory())
            mine, theirs = b, w
        else:
            b, w = play_game(opponent_factory(), agent_factory())
            mine, theirs = w, b
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            draws += 1
    return wins, draws, n


print("="*60)
print("بخش ۱: معادل بودن Minimax و Alpha-Beta (یک بازی کامل)")
print("="*60)
random.seed(1)
g = Othello(6)
mm = MinimaxAgent(depth=4); ab = AlphaBetaAgent(depth=4)
player = BLACK; mismatch = 0; steps = 0
while not g.game_over():
    moves = g.get_valid_moves(player)
    if moves:
        m1 = mm.minimax(g.copy(), 4, True, player)[0]
        a1 = ab.alphabeta(g.copy(), 4, float('-inf'), float('inf'), True, player)[0]
        if abs(m1 - a1) > 1e-6:
            mismatch += 1
        mv = ab.choose_move(g, player)
        g.make_move(player, *mv)
        steps += 1
    player = -player
print(f"تعداد گام‌های بررسی‌شده: {steps}, اختلاف مقدار: {mismatch} (انتظار: 0)")

print()
print("="*60)
print("بخش ۲: نرخ برد در برابر Random و Greedy (۲۰ بازی هرکدام)")
print("="*60)
N = 20
for name, depth in [("Minimax(d=3)", 3), ("Minimax(d=4)", 4)]:
    w, d, n = winrate(lambda dd=depth: MinimaxAgent(dd), RandomAgent, N)
    print(f"{name:16s} vs Random : برد={w:2d}/{n}  مساوی={d}  نرخ برد={100*w/n:.0f}%")
    w, d, n = winrate(lambda dd=depth: MinimaxAgent(dd), GreedyAgent, N)
    print(f"{name:16s} vs Greedy : برد={w:2d}/{n}  مساوی={d}  نرخ برد={100*w/n:.0f}%")

for name, depth in [("AlphaBeta(d=3)", 3), ("AlphaBeta(d=4)", 4)]:
    w, d, n = winrate(lambda dd=depth: AlphaBetaAgent(dd), RandomAgent, N)
    print(f"{name:16s} vs Random : برد={w:2d}/{n}  مساوی={d}  نرخ برد={100*w/n:.0f}%")
    w, d, n = winrate(lambda dd=depth: AlphaBetaAgent(dd), GreedyAgent, N)
    print(f"{name:16s} vs Greedy : برد={w:2d}/{n}  مساوی={d}  نرخ برد={100*w/n:.0f}%")

print()
print("="*60)
print("بخش ۳: مقایسه‌ی گره‌ها و زمان اجرا (یک بازی کامل، Minimax vs Random)")
print("="*60)
for depth in [2, 3, 4, 5]:
    # Minimax
    random.seed(5)
    g = Othello(6); player = BLACK
    mm = MinimaxAgent(depth); total_nodes_mm = 0; t0 = time.time()
    while not g.game_over():
        moves = g.get_valid_moves(player)
        if moves:
            mv = mm.choose_move(g, player); total_nodes_mm += mm.nodes
            g.make_move(player, *mv)
        player = -player
    t_mm = time.time() - t0
    # AlphaBeta
    random.seed(5)
    g = Othello(6); player = BLACK
    ab = AlphaBetaAgent(depth); total_nodes_ab = 0; t0 = time.time()
    while not g.game_over():
        moves = g.get_valid_moves(player)
        if moves:
            mv = ab.choose_move(g, player); total_nodes_ab += ab.nodes
            g.make_move(player, *mv)
        player = -player
    t_ab = time.time() - t0
    reduction = 100 * (1 - total_nodes_ab / total_nodes_mm) if total_nodes_mm else 0
    print(f"عمق {depth}: Minimax گره={total_nodes_mm:7d} زمان={t_mm:.3f}s | "
          f"AlphaBeta گره={total_nodes_ab:7d} زمان={t_ab:.3f}s | کاهش گره={reduction:.1f}%")

print()
print("="*60)
print("بخش ۴: تست روی صفحه‌ی غیر ۶×۶ (۸×۸) — طبق هشدار پروژه")
print("="*60)
random.seed(3)
b, w = play_game(AlphaBetaAgent(3), RandomAgent(), size=8)
print(f"AlphaBeta(d=3) [Black] vs Random [White] روی 8x8 → سیاه={b} سفید={w}")
