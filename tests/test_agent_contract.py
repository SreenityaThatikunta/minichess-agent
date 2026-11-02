"""What the match runner is entitled to assume: a legal move, every time,
inside the clock. A violation here forfeits a real game.
"""

import random
import time
import unittest

from board import GameEngine
from config import BLACK_KING, BLACK_ROOK, WHITE_KING, WHITE_ROOK
from minichess_agent import MiniChessAgent
from tests.helpers import engine_move_set, position


def packed_choice(engine, agent):
    """The agent's chosen move as a (from, to) pair, for set comparison."""
    move = agent.get_best_move()
    if move is None:
        return None
    return (move.start_row * 6 + move.start_col,
            move.end_row * 6 + move.end_col)


class ReturnsOnlyLegalMoves(unittest.TestCase):

    def test_opening_move_is_legal(self):
        engine = GameEngine()
        agent = MiniChessAgent(engine)
        self.assertIn(packed_choice(engine, agent), engine_move_set(engine))

    def test_stays_legal_through_a_full_self_play_game(self):
        """Both sides driven by the agent, so every position it can reach in
        its own play is covered rather than just the opening.
        """
        engine = GameEngine()
        white = MiniChessAgent(engine)
        black = MiniChessAgent(engine)
        white.FIXED_DEPTH = black.FIXED_DEPTH = 3
        plies = 0

        for _ in range(40):
            legal = engine_move_set(engine)
            if not legal:
                break
            agent = white if engine.white_to_move else black
            move = agent.get_best_move()
            self.assertIsNotNone(move, "returned None with moves available")
            as_pair = (move.start_row * 6 + move.start_col,
                       move.end_row * 6 + move.end_col)
            self.assertIn(as_pair, legal, f"illegal move at ply {plies}")
            engine.make_move(move)
            plies += 1

        self.assertGreater(plies, 10, "game ended too early to prove much")

    def test_returns_a_move_when_in_check(self):
        """Being in check narrows the move list; the agent must still answer."""
        engine = position({
            (0, 0): BLACK_KING,
            (0, 3): WHITE_ROOK,
            (7, 3): WHITE_KING,
        }, white_to_move=False)
        agent = MiniChessAgent(engine)
        agent.FIXED_DEPTH = 3
        self.assertIn(packed_choice(engine, agent), engine_move_set(engine))

    def test_returns_none_only_when_there_are_no_moves(self):
        """A mated side has nothing to play, and must not invent a move."""
        engine = position({
            (0, 0): BLACK_KING,
            (0, 5): WHITE_ROOK,      # mate along row 0
            (1, 5): WHITE_ROOK,      # covers the escape row
            (7, 3): WHITE_KING,
        }, white_to_move=False)
        self.assertEqual(engine.get_game_state(), "checkmate")

        agent = MiniChessAgent(engine)
        self.assertIsNone(agent.get_best_move())


class FindsForcedTactics(unittest.TestCase):

    def test_plays_the_only_mate_in_one(self):
        """The position has exactly one mating move, so the assertion is on a
        unique answer rather than on 'something reasonable'.
        """
        engine = position({
            (0, 0): BLACK_KING,
            (2, 5): WHITE_ROOK,
            (1, 4): WHITE_ROOK,
            (7, 3): WHITE_KING,
        })

        mates = []
        for move in engine.get_legal_moves():
            engine.make_move(move)
            if engine.get_game_state() == "checkmate":
                mates.append((move.start_row * 6 + move.start_col,
                              move.end_row * 6 + move.end_col))
            engine.undo_move()
        self.assertEqual(len(mates), 1, "fixture no longer has a unique mate")

        agent = MiniChessAgent(engine)
        agent.FIXED_DEPTH = 4
        self.assertEqual(packed_choice(engine, agent), mates[0])

    def test_takes_free_material(self):
        """An undefended rook is on offer and nothing else is close in value."""
        engine = position({
            (4, 2): WHITE_ROOK,
            (2, 2): BLACK_ROOK,      # undefended, straight up the file
            (0, 0): BLACK_KING,
            (7, 3): WHITE_KING,
        })
        agent = MiniChessAgent(engine)
        agent.FIXED_DEPTH = 4
        self.assertEqual(packed_choice(engine, agent), (4 * 6 + 2, 2 * 6 + 2))


class RespectsTheClock(unittest.TestCase):
    """A time forfeit scores zero regardless of how good the move was, so the
    budget matters more than the search depth.
    """

    def test_a_single_move_stays_inside_its_slice(self):
        engine = GameEngine()
        agent = MiniChessAgent(engine)

        start = time.perf_counter()
        agent.get_best_move()
        elapsed = time.perf_counter() - start

        # The agent paces a 60s game over at most 75 moves. A generous ceiling
        # here: the point is to catch a runaway search, not to benchmark a
        # loaded CI runner.
        self.assertLess(elapsed, 15.0,
                        f"one move took {elapsed:.1f}s of a 60s game")

    def test_spends_its_whole_budget_over_a_sequence_of_moves(self):
        """The clock is self-managed, so a sequence must not overrun 60s."""
        engine = GameEngine()
        agent = MiniChessAgent(engine)

        start = time.perf_counter()
        for _ in range(8):
            move = agent.get_best_move()
            if move is None:
                break
            engine.make_move(move)
            engine.make_move(random.Random(3).choice(engine.get_legal_moves()))
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 60.0,
                        f"8 moves consumed {elapsed:.1f}s of a 60s game")


if __name__ == "__main__":
    unittest.main()
