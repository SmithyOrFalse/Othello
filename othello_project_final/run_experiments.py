"""
Run experiments: win rates against Random and Greedy agents,
and comparison of node counts / execution time for Minimax vs Alpha-Beta.
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


def winrate(agent_factory, opponent_factory, n=20):
    """
    Play n games alternating colours so neither side has a systematic advantage.
    Returns (wins, draws, total).
    """
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


print("=" * 60)
print("Part 1: Equivalence check — Minimax vs Alpha-Beta (one full game)")
print("=" * 60)
random.seed(1)
g = Othello(6)
mm = MinimaxAgent(depth=4)
ab = AlphaBetaAgent(depth=4)
player = BLACK
mismatch = 0
steps = 0
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
print(f"Steps checked: {steps}, value mismatches: {mismatch} (expected: 0)")

print()
print("=" * 60)
print("Part 2: Win rates vs Random and Greedy (20 games each)")
print("=" * 60)
N = 20
for name, depth in [("Minimax(d=3)", 3), ("Minimax(d=4)", 4)]:
    w, d, n = winrate(lambda dd=depth: MinimaxAgent(dd), RandomAgent, N)
    print(f"{name:16s} vs Random : wins={w:2d}/{n}  draws={d}  win rate={100*w/n:.0f}%")
    w, d, n = winrate(lambda dd=depth: MinimaxAgent(dd), GreedyAgent, N)
    print(f"{name:16s} vs Greedy : wins={w:2d}/{n}  draws={d}  win rate={100*w/n:.0f}%")

for name, depth in [("AlphaBeta(d=3)", 3), ("AlphaBeta(d=4)", 4)]:
    w, d, n = winrate(lambda dd=depth: AlphaBetaAgent(dd), RandomAgent, N)
    print(f"{name:16s} vs Random : wins={w:2d}/{n}  draws={d}  win rate={100*w/n:.0f}%")
    w, d, n = winrate(lambda dd=depth: AlphaBetaAgent(dd), GreedyAgent, N)
    print(f"{name:16s} vs Greedy : wins={w:2d}/{n}  draws={d}  win rate={100*w/n:.0f}%")

print()
print("=" * 60)
print("Part 3: Node count and runtime comparison (one full game each depth)")
print("=" * 60)
for depth in [2, 3, 4, 5]:
    # Minimax
    random.seed(5)
    g = Othello(6)
    player = BLACK
    mm = MinimaxAgent(depth)
    total_nodes_mm = 0
    t0 = time.time()
    while not g.game_over():
        moves = g.get_valid_moves(player)
        if moves:
            mv = mm.choose_move(g, player)
            total_nodes_mm += mm.nodes
            g.make_move(player, *mv)
        player = -player
    t_mm = time.time() - t0

    # AlphaBeta
    random.seed(5)
    g = Othello(6)
    player = BLACK
    ab = AlphaBetaAgent(depth)
    total_nodes_ab = 0
    t0 = time.time()
    while not g.game_over():
        moves = g.get_valid_moves(player)
        if moves:
            mv = ab.choose_move(g, player)
            total_nodes_ab += ab.nodes
            g.make_move(player, *mv)
        player = -player
    t_ab = time.time() - t0

    reduction = 100 * (1 - total_nodes_ab / total_nodes_mm) if total_nodes_mm else 0
    print(f"Depth {depth}: Minimax nodes={total_nodes_mm:7d} time={t_mm:.3f}s | "
          f"AlphaBeta nodes={total_nodes_ab:7d} time={t_ab:.3f}s | reduction={reduction:.1f}%")

print()
print("=" * 60)
print("Part 4: Test on non-6x6 board (8x8) — as noted in project spec")
print("=" * 60)
random.seed(3)
b, w = play_game(AlphaBetaAgent(3), RandomAgent(), size=8)
print(f"AlphaBeta(d=3) [Black] vs Random [White] on 8x8 -> Black={b}  White={w}")
