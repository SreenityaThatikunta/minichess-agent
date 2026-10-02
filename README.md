# 6×8 Minichess Agent

An adversarial search agent for a 6×8 chess variant. It reaches roughly **13 ply
in a 60-second game in pure Python**, and scores the maximum possible result
against every opponent it was tested against.

Single file, standard library only.

## Results

| Test | Result |
| --- | --- |
| vs a random-move opponent | 600 points a game, 4–0, checkmate every game |
| vs a random opponent with think delay | 600 points a game, 4–0, checkmate every game |
| vs this agent's own earlier version | 600 points a game, 4–0, checkmate every game |
| Search depth on the clock | 13.35 ply |
| Node rate | ~290,000 positions/second |
| Time forfeits | zero, across every game played |

600 is the maximum the rules allow.

## The interesting part

The game's scoring is **not zero-sum**. A match pays 600 for checkmate, 0 for a
loss, and otherwise the value of the pieces *you* captured — your opponent's
captures never subtract from your score. Total capturable material is 530, less
than the 600 a mate pays.

So the priority order is: never lose, then mate, then capture. That drives every
design decision, and my first instinct about how to exploit it — trade very
freely, since even trades are pure profit — was measurably wrong. It scored 291
points a game against 515 for near-symmetric play, because bleeding material to
bank captures just gets you mated.

## How it works

- Iterative-deepening negamax with alpha-beta and principal-variation search,
  so it can stop on the clock at any point and still return a usable move
- Transposition table on a 64-bit Zobrist hash, killer moves, history heuristic
- Null-move pruning, late-move reductions, check extensions
- Quiescence search over captures, with static exchange evaluation for pruning
- Tapered piece-square tables plus rook file control, rook mobility and a king
  pawn shield
- Its own move generator — a 48-square array with precomputed attack tables and
  incremental make/unmake, **47× the node rate** of driving the engine's own
  generator, verified move-for-move against it across 217,000 positions
- A 379-position opening book, and a clock allocator that cannot overrun

## Running it

```bash
python3 game_runner.py
```

Pits two agents against each other with the 60-second clock and prints the
board, move log and running score. Edit the `run_game(...)` call at the bottom
of that file to choose which agents play.

## Layout

| File | What it is |
| --- | --- |
| `minichess_agent.py` | The agent |
| `baseline_agent.py` | The earlier version, kept as a measurement baseline |
| `ENGINEERING.md` | Full write-up: design, measurements, what failed |
| `board.py`, `config.py`, `game_runner.py` | The game engine and runner. Third party — see `NOTICE` |
| `p22cs201.py`, `p25cs0004.py` | Sample random-move opponents. Third party |

## Full write-up

**[ENGINEERING.md](ENGINEERING.md)** has the detail: the complete architecture,
every approach that was tried and measured, the four predictions the data
contradicted, two real bugs and how they were caught, the testing methodology,
and the agent's known limitations.

## Licence

MIT, see `LICENSE`. It covers the agent and the write-up; the game engine and
sample opponents are third party and keep their original terms, set out in
`NOTICE`.
