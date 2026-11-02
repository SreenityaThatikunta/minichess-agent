"""Shared fixtures for the test suite.

Positions are built by writing the board array directly, since the engine has
no FEN parser. Row 0 is black's back rank, row 7 is white's.
"""

from board import GameEngine
from config import BOARD_HEIGHT, BOARD_WIDTH, EMPTY_SQUARE
from minichess_agent import MiniChessAgent


def empty_board():
    return [[EMPTY_SQUARE] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]


def position(cells, white_to_move=True):
    """An engine holding exactly `cells`, a {(row, col): piece} mapping."""
    engine = GameEngine()
    board = empty_board()
    for (row, col), piece in cells.items():
        board[row][col] = piece
    engine.board = board
    engine.white_to_move = white_to_move
    engine.move_log = []
    engine.position_history = {}
    engine.update_position_history()
    return engine


def synced_agent(engine):
    agent = MiniChessAgent(engine)
    agent._sync()
    return agent


def square(packed):
    """The (row, col) pair a packed square index refers to."""
    return divmod(packed, BOARD_WIDTH)


def engine_move_set(engine):
    """The engine's legal moves as a set of (from, to) square-index pairs."""
    return {
        (m.start_row * BOARD_WIDTH + m.start_col,
         m.end_row * BOARD_WIDTH + m.end_col)
        for m in engine.get_legal_moves()
    }


def agent_move_set(agent):
    """The same, from the agent's own generator.

    `_gen` is pseudo-legal by design — the search filters moves that leave the
    king attacked after making them, so the filter is reproduced here.
    """
    us = agent.stm
    legal = set()
    for packed in agent._gen():
        agent._make(packed)
        if not agent._attacked(agent.ksq[us], us == 0):
            legal.add((packed & 63, (packed >> 6) & 63))
        agent._undo(packed)
    return legal


def perft(engine, depth):
    moves = engine.get_legal_moves()
    if depth == 1:
        return len(moves)
    total = 0
    for move in moves:
        engine.make_move(move)
        total += perft(engine, depth - 1)
        engine.undo_move()
    return total
