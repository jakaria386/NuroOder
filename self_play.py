"""
==========================================================
self_play.py
Author : Md. Jakaria Masud

Project : NeuroOrder

Generate Self-Play Dataset
(Leakage-Free Version with game_id)
==========================================================
"""

import csv
import math
import random

from game_engine import (
    create_board,
    get_valid_moves,
    make_move,
    check_winner,
    minimax_ab
)

from features import extract_features


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

TOTAL_GAMES = 5000
SEARCH_DEPTH = 4

OUTPUT_FILE = "self_play_data.csv"


# ----------------------------------------------------
# Label Generation
#
# Smaller search tree
# = Better move
# ----------------------------------------------------

def generate_label(board, move, player):

    next_board = make_move(board, move, player)

    opponent = 2 if player == 1 else 1

    _, nodes = minimax_ab(
        next_board,
        SEARCH_DEPTH - 1,
        -math.inf,
        math.inf,
        opponent
    )

    if nodes == 0:
        return 1.0

    return round(1 / nodes, 6)


# ----------------------------------------------------
# Save all candidate moves
# ----------------------------------------------------

def save_candidates(board, player, writer, game_id):

    valid_moves = get_valid_moves(board)

    for move in valid_moves:

        features = extract_features(
            board,
            move,
            player
        )

        label = generate_label(
            board,
            move,
            player
        )

        writer.writerow([

            game_id,

            features[0],

            features[1],

            features[2],

            features[3],

            features[4],

            label

        ])


# ----------------------------------------------------
# Play One Game
# ----------------------------------------------------

def play_game(writer, game_id):

    board = create_board()

    player = 1

    winner = None

    while winner is None:

        valid_moves = get_valid_moves(board)

        if len(valid_moves) == 0:
            break

        # Save all candidate moves
        save_candidates(
            board,
            player,
            writer,
            game_id
        )

        # 20% Random Opening
        if random.random() < 0.20:

            move = random.choice(valid_moves)

        # 80% Alpha-Beta
        else:

            move, _ = minimax_ab(

                board,

                SEARCH_DEPTH,

                -math.inf,

                math.inf,

                player

            )

        board = make_move(
            board,
            move,
            player
        )

        winner = check_winner(board)

        player = 2 if player == 1 else 1


# ----------------------------------------------------
# Generate Dataset
# ----------------------------------------------------

def generate_dataset():

    with open(

        OUTPUT_FILE,

        "w",

        newline=""

    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "game_id",

            "piece_difference",

            "center_control",

            "player_threats",

            "opponent_threats",

            "mobility",

            "label"

        ])

        for game_id in range(TOTAL_GAMES):

            play_game(
                writer,
                game_id
            )

            if (game_id + 1) % 100 == 0:

                print(

                    f"{game_id + 1} Games Completed..."

                )

    print("\nDataset Generated Successfully.")

    print(f"Saved as {OUTPUT_FILE}")


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":

    generate_dataset()