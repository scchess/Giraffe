#!/usr/bin/env python3

import chess
import sys

from chess import pgn

if len(sys.argv) != 4:
    print("Usage:", sys.argv[0], "<PGN file> <start position after # of moves> <unique only?>")
    sys.exit(1)

if sys.argv[1] == '-':
    pgn = sys.stdin
else:
    pgn = open(sys.argv[1])

unique_only = (int(sys.argv[3]) != 0)

numSkip = int(sys.argv[2])

positionsSeen = set()

game = chess.pgn.read_game(pgn)

while game:
    node = game
    for i in range(numSkip):
        if not node.variations:
            break
        node = node.variations[0]

    if node.variations:
        epd = node.board().epd()
        if not unique_only or not epd in positionsSeen:
            print(node.board().epd())
            positionsSeen.add(epd)

    game = chess.pgn.read_game(pgn)
