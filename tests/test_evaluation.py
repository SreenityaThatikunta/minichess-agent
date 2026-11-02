"""Static exchange evaluation and the evaluation's sign convention.

The scoring in this variant is not zero-sum — only the material you capture
earns points — so the evaluation is deliberately asymmetric. These tests pin
the parts of that asymmetry a refactor could silently invert.
"""

import unittest

from config import (BLACK_KING, BLACK_PAWN, BLACK_ROOK, WHITE_KING,
                    WHITE_PAWN, WHITE_ROOK)
from tests.helpers import position, synced_agent


def capture_see(agent, target):
    """The SEE score of the one generated capture landing on `target`."""
    scores = [agent._see(mv) for mv in agent._gen(caps_only=True)
              if ((mv >> 6) & 63) == target]
    assert len(scores) == 1, f"expected one capture of {target}, got {scores}"
    return scores[0]


class StaticExchangeEvaluation(unittest.TestCase):
    """SEE decides which captures the quiescence search is allowed to skip.
    An inverted sign here makes the agent hang material on purpose.
    """

    def test_capturing_an_undefended_rook_scores_its_value(self):
        engine = position({
            (4, 2): WHITE_ROOK,
            (2, 2): BLACK_ROOK,
            (0, 0): BLACK_KING,
            (7, 3): WHITE_KING,
        })
        agent = synced_agent(engine)
        self.assertEqual(capture_see(agent, 2 * 6 + 2), 100)

    def test_rook_taking_a_defended_pawn_is_a_loss(self):
        """Rook (100) for a pawn (20), then recaptured: a losing exchange."""
        engine = position({
            (4, 2): WHITE_ROOK,
            (2, 2): BLACK_PAWN,
            (1, 1): BLACK_PAWN,      # recaptures on (2, 2)
            (0, 0): BLACK_KING,
            (7, 3): WHITE_KING,
        })
        agent = synced_agent(engine)
        self.assertEqual(capture_see(agent, 2 * 6 + 2), 20 - 100)

    def test_an_even_trade_is_not_scored_as_a_loss(self):
        engine = position({
            (4, 2): WHITE_ROOK,
            (2, 2): BLACK_ROOK,
            (1, 1): BLACK_PAWN,      # recaptures, so rook for rook
            (0, 0): BLACK_KING,
            (7, 3): WHITE_KING,
        })
        agent = synced_agent(engine)
        self.assertGreaterEqual(capture_see(agent, 2 * 6 + 2), 0)


class EvaluationSign(unittest.TestCase):

    def test_material_advantage_favours_the_side_holding_it(self):
        """`_eval_stm` is from the side to move's point of view, so the same
        position must read positive for the player who is up a rook.
        """
        cells = {
            (4, 2): WHITE_ROOK,
            (6, 2): WHITE_PAWN,
            (1, 2): BLACK_PAWN,
            (0, 0): BLACK_KING,
            (7, 3): WHITE_KING,
        }
        white_view = synced_agent(position(cells, white_to_move=True))
        black_view = synced_agent(position(cells, white_to_move=False))

        self.assertGreater(white_view._eval_stm(), 0)
        self.assertLess(black_view._eval_stm(), 0)

    def test_a_balanced_position_is_near_level(self):
        cells = {
            (4, 2): WHITE_ROOK,
            (3, 2): BLACK_ROOK,
            (6, 1): WHITE_PAWN,
            (1, 1): BLACK_PAWN,
            (0, 0): BLACK_KING,
            (7, 0): WHITE_KING,
        }
        agent = synced_agent(position(cells))
        # Mirror-image material and pawns: any edge is positional, not a
        # material miscount, so it should be small next to a pawn's 20.
        self.assertLess(abs(agent._eval_stm()), 60)


if __name__ == "__main__":
    unittest.main()
