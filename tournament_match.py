#!/usr/bin/env python3
"""Play matches under the tournament scoring the agent was built against.

`game_runner.py` is the sample runner this variant shipped with, and it scores
games additively: both sides keep their capture points, a check is worth +2 and
a checkmate adds +300. The tournament rules are different and are what every
number in README.md and ENGINEERING.md was measured under:

  * Checkmate pays an absolute 600, overriding any captured material.
  * A loss pays 0, regardless of what was captured first.
  * A draw or stalemate pays the value of the pieces you captured.
  * Exceeding the 60-second clock loses the game outright.

Those two rule sets reward completely different play, which is the whole
subject of ENGINEERING.md, so this runner exists to make the published results
reproducible:

    python3 tournament_match.py --games 4

Run `--help` for the full set of options.
"""

import argparse
import importlib
import time

from board import GameEngine
from config import PIECE_VALUES

# The tournament pays a flat 600 for a mate and nothing for a loss.
MATE_SCORE = 600
LOSS_SCORE = 0

# Clock and length limits from the rules.
TIME_LIMIT_SECONDS = 60.0
MAX_MOVES_PER_SIDE = 75

# Where the stock agents live, so --white/--black can name them directly.
KNOWN_AGENTS = {
    "agent": ("minichess_agent", "MiniChessAgent"),
    "baseline": ("baseline_agent", "BaselineAgent"),
    "random": ("p25cs0004", "P25CS0004"),
    "random-delay": ("p22cs201", "P22CS201"),
}


def load_agent(name):
    """Resolve an agent name, either a known alias or `module:ClassName`."""
    if name in KNOWN_AGENTS:
        module_name, class_name = KNOWN_AGENTS[name]
    elif ":" in name:
        module_name, class_name = name.split(":", 1)
    else:
        raise SystemExit(
            f"unknown agent {name!r}; use one of {sorted(KNOWN_AGENTS)} "
            f"or module:ClassName"
        )
    return getattr(importlib.import_module(module_name), class_name)


class Result:
    """One game's outcome, scored under the tournament rules."""

    def __init__(self, white_name, black_name):
        self.white_name = white_name
        self.black_name = black_name
        self.captures = {"white": 0, "black": 0}
        self.elapsed = {"white": 0.0, "black": 0.0}
        self.plies = 0
        self.outcome = "unfinished"
        self.winner = None

    def score(self, side):
        """The final score for `side`, applying the overrides in rule order."""
        if self.winner is not None:
            return MATE_SCORE if self.winner == side else LOSS_SCORE
        # Draw, stalemate, repetition or the move limit: captures only. Checks
        # are worth +2 in the points table but the rules settle a drawn game
        # "exclusively from the points earned by capturing pieces", so they are
        # deliberately not counted here.
        return self.captures[side]

    @property
    def white_score(self):
        return self.score("white")

    @property
    def black_score(self):
        return self.score("black")

    def summary(self):
        detail = self.outcome
        if self.winner:
            detail = f"{self.outcome} ({self.winner} wins)"
        return (f"{self.white_name} {self.white_score:>4}  -  "
                f"{self.black_score:<4} {self.black_name}   "
                f"[{self.plies} plies, {detail}, "
                f"clock {self.elapsed['white']:.1f}s / "
                f"{self.elapsed['black']:.1f}s]")


def play_game(white_type, black_type, verbose=False):
    engine = GameEngine()
    white = white_type(engine)
    black = black_type(engine)
    result = Result(white_type.__name__, black_type.__name__)

    while True:
        side = "white" if engine.white_to_move else "black"
        mover = white if engine.white_to_move else black

        state = engine.get_game_state()
        if state == "checkmate":
            # The side to move is mated, so the other side delivered it.
            result.outcome = "checkmate"
            result.winner = "black" if side == "white" else "white"
            break
        if state == "stalemate":
            result.outcome = "stalemate"
            break
        if engine.get_repetition_count() >= 3:
            result.outcome = "draw by repetition"
            break
        if result.plies >= MAX_MOVES_PER_SIDE * 2:
            result.outcome = "move limit"
            break

        started = time.perf_counter()
        move = mover.get_best_move()
        result.elapsed[side] += time.perf_counter() - started

        if result.elapsed[side] > TIME_LIMIT_SECONDS:
            # Overrunning the clock loses the game however good the position.
            result.outcome = "time forfeit"
            result.winner = "black" if side == "white" else "white"
            break

        if move is None:
            result.outcome = "no move returned"
            result.winner = "black" if side == "white" else "white"
            break

        captured = move.piece_captured
        if captured and captured in PIECE_VALUES:
            result.captures[side] += abs(PIECE_VALUES[captured])

        engine.make_move(move)
        result.plies += 1

        if verbose:
            print(f"  {result.plies:>3}. {side:<5} {move}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--white", default="agent",
                        help="white agent (default: agent)")
    parser.add_argument("--black", default="baseline",
                        help="black agent (default: baseline)")
    parser.add_argument("--games", type=int, default=1,
                        help="games to play, alternating colours (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="print every move")
    args = parser.parse_args()

    first, second = load_agent(args.white), load_agent(args.black)
    totals = {first.__name__: 0, second.__name__: 0}
    wins = {first.__name__: 0, second.__name__: 0}

    for game in range(args.games):
        # Alternate colours so neither agent keeps the first-move advantage.
        white_type, black_type = (first, second) if game % 2 == 0 \
            else (second, first)
        result = play_game(white_type, black_type, verbose=args.verbose)

        totals[white_type.__name__] += result.white_score
        totals[black_type.__name__] += result.black_score
        if result.winner == "white":
            wins[white_type.__name__] += 1
        elif result.winner == "black":
            wins[black_type.__name__] += 1

        print(f"game {game + 1}: {result.summary()}")

    if args.games > 1:
        print("\ntotals over", args.games, "games")
        for name in (first.__name__, second.__name__):
            average = totals[name] / args.games
            print(f"  {name:<18} {totals[name]:>5} points "
                  f"({average:.1f} a game, {wins[name]} wins)")


if __name__ == "__main__":
    main()
