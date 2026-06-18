"""
University: University of Isfahan
Faculty: Mathematics and Statistics
Department: Computer Science
Course: Artificial Intelligence
Professor: Dr. Faria Nasiri Mofakham
TAs: MehrAzin Marzough, Mohammad Karimi, Anahita Honarmandian
Project: Adversarial Search in Othello (Minimax and Alpha-Beta Pruning)
"""

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from tournament import play_game


def fmt(score):
    b, w = score
    if b > w:
        return f"Black wins  (Black={b}, White={w})"
    elif w > b:
        return f"White wins  (Black={b}, White={w})"
    return f"Draw  (Black={b}, White={w})"


if __name__ == "__main__":
    print("Greedy (Black) vs Random (White):")
    print("  ", fmt(play_game(GreedyAgent(), RandomAgent())))

    print("\nMinimax d=4 (Black) vs Random (White):")
    print("  ", fmt(play_game(MinimaxAgent(depth=4), RandomAgent())))

    print("\nMinimax d=4 (Black) vs Greedy (White):")
    print("  ", fmt(play_game(MinimaxAgent(depth=4), GreedyAgent())))

    print("\nAlphaBeta d=4 (Black) vs Greedy (White):")
    print("  ", fmt(play_game(AlphaBetaAgent(depth=4), GreedyAgent())))

    print("\nAlphaBeta d=4 (Black) vs Minimax d=4 (White):")
    print("  ", fmt(play_game(AlphaBetaAgent(depth=4), MinimaxAgent(depth=4))))
