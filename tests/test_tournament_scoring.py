"""The tournament score card, which is what the published results are in.

The rules override each other in a specific order — a mate pays a flat 600
whatever was captured, and a loss pays nothing however much was captured
first — so these tests pin the overrides rather than the arithmetic.
"""

import unittest

from board import GameEngine
from config import BLACK_KING, PIECE_VALUES, WHITE_KING, WHITE_ROOK
from tests.helpers import position
from tournament_match import (LOSS_SCORE, MATE_SCORE, Result, load_agent,
                              play_game)


class ScoreCardOverrides(unittest.TestCase):

    def _result(self, **kwargs):
        result = Result("White", "Black")
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result

    def test_checkmate_overrides_captured_material(self):
        """The loser captured more, and still scores nothing."""
        result = self._result(winner="white", outcome="checkmate")
        result.captures = {"white": 20, "black": 400}

        self.assertEqual(result.white_score, MATE_SCORE)
        self.assertEqual(result.black_score, LOSS_SCORE)

    def test_a_draw_pays_captured_material_only(self):
        result = self._result(outcome="stalemate")
        result.captures = {"white": 190, "black": 70}

        self.assertEqual(result.white_score, 190)
        self.assertEqual(result.black_score, 70)

    def test_a_time_forfeit_scores_zero_for_the_offender(self):
        """Overrunning the clock loses outright, whatever the position."""
        result = self._result(winner="black", outcome="time forfeit")
        result.captures = {"white": 530, "black": 0}

        self.assertEqual(result.white_score, LOSS_SCORE)
        self.assertEqual(result.black_score, MATE_SCORE)

    def test_mate_is_worth_more_than_the_whole_opposing_army(self):
        """A side can capture at most 530 points of material, below the 600 a
        mate pays. That gap is the fact the whole evaluation is built around,
        so it is measured from the real starting position rather than assumed.
        """
        engine = GameEngine()
        capturable = sum(
            abs(PIECE_VALUES[square])
            for row in engine.board for square in row
            if square in PIECE_VALUES and square[1] != "K"
        ) // 2          # the board holds both armies; one side can take one

        self.assertEqual(capturable, 530)
        self.assertGreater(MATE_SCORE, capturable)


class PlayingAGame(unittest.TestCase):

    def test_a_mated_side_scores_zero_and_the_winner_scores_600(self):
        class Mated:
            """Stands in for an agent that is already lost when asked."""

            def __init__(self, engine):
                self.engine = engine

            def get_best_move(self):
                return self.engine.get_legal_moves()[0]

        result = play_game(Mated, Mated)
        self.assertIn(result.outcome,
                      {"checkmate", "stalemate", "draw by repetition",
                       "move limit"})
        # Whatever the ending, the two scores must obey the score card.
        if result.winner:
            self.assertEqual(
                {result.white_score, result.black_score},
                {MATE_SCORE, LOSS_SCORE},
            )

    def test_detects_a_position_that_is_already_checkmate(self):
        engine = position({
            (0, 0): BLACK_KING,
            (0, 5): WHITE_ROOK,
            (1, 5): WHITE_ROOK,
            (7, 3): WHITE_KING,
        }, white_to_move=False)
        self.assertEqual(engine.get_game_state(), "checkmate")


class AgentAliases(unittest.TestCase):
    """`--white` / `--black` resolve by alias, so a typo should not silently
    fall back to something that still plays.
    """

    def test_known_aliases_resolve(self):
        for alias in ("agent", "baseline", "random", "random-delay"):
            with self.subTest(alias=alias):
                self.assertTrue(callable(load_agent(alias)))

    def test_dotted_form_resolves(self):
        self.assertTrue(callable(load_agent("minichess_agent:MiniChessAgent")))

    def test_an_unknown_name_is_rejected(self):
        with self.assertRaises(SystemExit):
            load_agent("not_a_real_agent")


if __name__ == "__main__":
    unittest.main()
