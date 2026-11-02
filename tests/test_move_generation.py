"""The agent carries its own move generator for speed, so it has to agree with
the engine's exactly. Any divergence is a move the search believes in but the
referee will reject, which is why this is the test that matters most.
"""

import random
import unittest

from board import GameEngine
from config import (BLACK_KING, BLACK_PAWN, BLACK_ROOK, WHITE_KING,
                    WHITE_KNIGHT, WHITE_ROOK)
from tests.helpers import (agent_move_set, engine_move_set, perft, position,
                           synced_agent)


class MoveGeneratorParity(unittest.TestCase):
    """The agent's generator against the engine's, position by position."""

    def _walk_random_game(self, seed, max_plies=40):
        """Plays a random game, comparing both generators at every position.

        Returns the number of positions compared. Random play mates quickly on
        a board this small, so breadth comes from many games, not long ones.
        """
        rng = random.Random(seed)
        engine = GameEngine()
        agent = synced_agent(engine)
        compared = 0

        for _ in range(max_plies):
            agent._sync()
            expected = engine_move_set(engine)
            actual = agent_move_set(agent)
            self.assertEqual(
                actual, expected,
                f"seed {seed}, ply {compared}: "
                f"agent-only {sorted(actual - expected)}, "
                f"engine-only {sorted(expected - actual)}"
            )
            compared += 1

            legal = engine.get_legal_moves()
            if not legal:
                break
            engine.make_move(rng.choice(legal))

        return compared

    def test_parity_over_many_random_games(self):
        total = sum(self._walk_random_game(seed) for seed in range(120))
        # Guards against the games all mating immediately and the assertions
        # above never really running.
        self.assertGreater(total, 500)

    def test_parity_in_check_positions(self):
        """Check evasions are the case most likely to diverge."""
        engine = position({
            (0, 0): BLACK_KING,
            (0, 3): WHITE_ROOK,      # black king is in check along row 0
            (7, 3): WHITE_KING,
            (4, 4): WHITE_KNIGHT,
        }, white_to_move=False)
        agent = synced_agent(engine)

        self.assertTrue(engine.is_in_check())
        self.assertEqual(agent_move_set(agent), engine_move_set(engine))

    def test_parity_in_pinned_positions(self):
        """A piece pinned against its own king may not move off the line."""
        engine = position({
            (0, 3): BLACK_KING,
            (2, 3): BLACK_ROOK,      # pinned down the file
            (6, 3): WHITE_ROOK,
            (7, 0): WHITE_KING,
        }, white_to_move=False)
        agent = synced_agent(engine)

        self.assertEqual(agent_move_set(agent), engine_move_set(engine))


class MakeUnmakeIsReversible(unittest.TestCase):
    """The generator's speed comes from incremental make/unmake rather than
    rebuilding state per node, so a leak here corrupts the whole search.
    Both bugs found during development lived in exactly this path.
    """

    def _snapshot(self, agent):
        return (
            list(agent.bd),
            agent.hash,
            list(agent.mat),
            agent.stm,
            list(agent.ksq),
            sorted(agent.occ[0]),
            sorted(agent.occ[1]),
            list(agent.npawn),
            [list(agent.pfile[0]), list(agent.pfile[1])],
            sorted(agent.rooks[0]),
            sorted(agent.rooks[1]),
            list(agent.pmg),
            list(agent.peg),
        )

    def test_every_move_restores_state_exactly(self):
        rng = random.Random(11)
        engine = GameEngine()
        agent = synced_agent(engine)
        checked = 0

        for _ in range(30):
            agent._sync()
            before = self._snapshot(agent)

            for packed in agent._gen():
                agent._make(packed)
                agent._undo(packed)
                self.assertEqual(
                    self._snapshot(agent), before,
                    f"state not restored after make/undo of {packed:#x}"
                )
                checked += 1

            legal = engine.get_legal_moves()
            if not legal:
                break
            engine.make_move(rng.choice(legal))

        self.assertGreater(checked, 100)

    def test_hash_returns_to_its_starting_value(self):
        """Zobrist keys drive the transposition table; a drifting hash hands
        back evaluations belonging to unrelated positions.
        """
        engine = GameEngine()
        agent = synced_agent(engine)
        start = agent.hash

        for packed in agent._gen():
            agent._make(packed)
            self.assertNotEqual(agent.hash, start,
                                "hash unchanged after a move")
            agent._undo(packed)
            self.assertEqual(agent.hash, start)


class Perft(unittest.TestCase):
    """Fixed node counts from the opening position. These pin the engine's
    own generator, so a change in the third-party board.py cannot quietly
    alter the rules the agent was tuned against.
    """

    EXPECTED = {1: 7, 2: 49, 3: 462, 4: 4351}

    def test_node_counts_from_the_initial_position(self):
        engine = GameEngine()
        for depth, expected in sorted(self.EXPECTED.items()):
            with self.subTest(depth=depth):
                self.assertEqual(perft(engine, depth), expected)

    def test_opening_position_is_restored_after_perft(self):
        engine = GameEngine()
        before = [row[:] for row in engine.board]
        perft(engine, 3)
        self.assertEqual(engine.board, before)
        self.assertTrue(engine.white_to_move)


if __name__ == "__main__":
    unittest.main()
