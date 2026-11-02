#!/usr/bin/env python3
"""Render a terminal capture into the SVG used as the README's screenshot.

Run from the repository root:

    python3 tools/render_demo.py

It plays a real game between the agent and the baseline, captures the board as
the runner prints it, and writes docs/demo.svg. Committing the generator keeps
the screenshot reproducible rather than a stale image nobody can regenerate.
"""

import io
import os
import sys
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_agent import BaselineAgent          # noqa: E402
from board import GameEngine                      # noqa: E402
from minichess_agent import MiniChessAgent         # noqa: E402

# A dark terminal palette reads correctly against both GitHub themes, which is
# why the capture is not generated twice for light and dark.
BACKGROUND = "#0d1117"
FOREGROUND = "#c9d1d9"
ACCENT = "#58a6ff"
MUTED = "#8b949e"
CHROME = "#30363d"

CHAR_WIDTH = 8.4
LINE_HEIGHT = 18
PADDING = 18
TITLE_BAR = 34


def capture_position(plies, depth, seconds_per_move):
    """Play `plies` real moves and return the board as the runner prints it.

    The capture runs at a fixed depth rather than on the real clock so that
    regenerating it takes seconds and lands on the same position each time.
    The clocks shown are the time actually consumed, not a placeholder.
    """
    import game_runner

    engine = GameEngine()
    white = MiniChessAgent(engine)
    black = BaselineAgent(engine)

    white.FIXED_DEPTH = depth
    black.time_limit = seconds_per_move

    spent = {True: 0.0, False: 0.0}
    for _ in range(plies):
        white_to_move = engine.white_to_move
        agent = white if white_to_move else black
        started = time.perf_counter()
        move = agent.get_best_move()
        spent[white_to_move] += time.perf_counter() - started
        if move is None:
            break
        engine.make_move(move)

    clock = game_runner.PlayerClock(60 - spent[True], 60 - spent[False])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        game_runner.display_board(engine, clock, white, black)
    return buffer.getvalue().strip("\n").splitlines(), white, depth


def escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


def render(lines, caption):
    body = list(lines) + ["", caption]
    width = int(max(len(line) for line in body) * CHAR_WIDTH) + PADDING * 2
    height = TITLE_BAR + len(body) * LINE_HEIGHT + PADDING

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
        f'font-size="13" role="img" '
        f'aria-label="A minichess game in progress in the terminal">',
        f'<rect width="{width}" height="{height}" rx="8" fill="{BACKGROUND}"/>',
        f'<rect width="{width}" height="{TITLE_BAR}" rx="8" fill="{BACKGROUND}"/>',
        f'<line x1="0" y1="{TITLE_BAR}" x2="{width}" y2="{TITLE_BAR}" '
        f'stroke="{CHROME}" stroke-width="1"/>',
    ]
    for index, colour in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        out.append(f'<circle cx="{18 + index * 18}" cy="17" r="6" '
                   f'fill="{colour}"/>')
    out.append(f'<text x="{width / 2}" y="22" fill="{MUTED}" '
               f'text-anchor="middle" font-size="12">python3 '
               f'game_runner.py</text>')

    y = TITLE_BAR + PADDING + 2
    for line in body:
        colour = FOREGROUND
        if line is body[-1]:
            colour = ACCENT
        out.append(f'<text x="{PADDING}" y="{y}" fill="{colour}" '
                   f'xml:space="preserve">{escape(line)}</text>')
        y += LINE_HEIGHT
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    lines, agent, depth = capture_position(plies=12, depth=6,
                                           seconds_per_move=0.1)
    caption = (f"capture pinned to depth {depth} for reproducibility "
               f"- {agent.nodes_expanded:,} nodes on the last move")
    svg = render(lines, caption)

    target = os.path.join("docs", "demo.svg")
    with open(target, "w") as handle:
        handle.write(svg)
    print(f"wrote {target} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
