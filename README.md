# 6×8 Minichess Agent

An adversarial search agent for a 6×8 chess variant. It reaches roughly **13 ply
in a 60-second game in pure Python**, and scores the maximum possible result
against every opponent it was tested against.

The interesting part of this project isn't the search — alpha-beta is
well-trodden. It's that the game's scoring rules are **not zero-sum**, which
changes what the evaluation function should optimise, and that my first instinct
about how to exploit that was measurably wrong.

Everything below was measured rather than assumed. Where the reasoning turned
out to be wrong, that's written down too, because it's most of what happened.

---

## The game

A chess variant on a 6-wide, 8-deep board, 12 pieces a side: one knight, two
bishops, two rooks, a king and six pawns. All pieces move as in standard chess,
with these removals:

- No pawn promotion
- No castling
- No en passant
- No double pawn push — pawns move one square only

Each side gets a 60-second clock and there's a hard limit of 150 total turns, 75
a side. Exceeding the clock loses the match outright.

## The scoring, which is the whole problem

This is what makes it not-chess. A match is worth:

| Outcome | Your score |
| --- | --- |
| Checkmate | 600, absolute — it overrides captured material |
| Loss | 0, regardless of what you captured |
| Draw, stalemate, or turn limit | the value of the pieces you captured |

Capture values: pawn 20, bishop 70, knight 70, rook 100.

Two consequences fall straight out of this:

1. **Your opponent's captures never subtract from your score.** The game is not
   zero-sum. An even trade banks points for you and costs you nothing directly.
2. **Total capturable material is 530, which is less than the 600 a mate pays.**
   So mating always beats farming, and losing wipes out everything you earned.

That ordering — never lose, then mate, then capture — drives every design
decision in the agent.

---

## Results

| Test | Result |
| --- | --- |
| vs a random-move opponent | 600 points a game, 4–0, checkmate every game |
| vs a random opponent with think delay | 600 points a game, 4–0, checkmate every game |
| vs this agent's own earlier version | 600 points a game, 4–0, checkmate every game |
| vs its immediate predecessor | tie: −19.5 ± 62.4, 0.31σ, n=20 |
| Search depth on the clock | 13.35 ply |
| Node rate | ~290,000 positions/second |
| Time forfeits | zero, across every game played |

600 is the maximum the rules allow. Against its own immediate predecessor it's a
statistical tie — the error bars are too wide to resolve a 30-point difference
at twenty games, which would need roughly 400 games or about thirteen hours of
wall clock. So the final round of work is best described as **verified safe and
unproven better**; what definitely improved is the foundation underneath it.

---

## How it works

### Search

Iterative-deepening negamax with alpha-beta, so it can stop on the clock at any
point and still return the best move from the last completed iteration. On top
of that:

- Principal-variation search
- Transposition table keyed on a 64-bit Zobrist hash, 400,000 entries
- Killer moves, two per ply, plus a history heuristic indexed by from-and-to
  square
- Null-move pruning, guarded against zugzwang by requiring non-pawn material
  and never firing while in check
- Late-move reductions on quiet moves
- Check extensions, bounded so perpetual-check lines can't recurse forever
- Quiescence search over captures with full check evasions, SEE pruning and
  delta pruning
- Aspiration windows at the root

### Evaluation

Scored from the root player's point of view, in points:

- Material at the game's own capture values, so the search optimises the real
  objective rather than a proxy
- Tapered piece-square tables — middlegame and endgame blended by remaining
  material, maintained incrementally so a leaf evaluation is constant-time
- Rook open-file bonus, rook mobility, and a phase-scaled king pawn shield
- An endgame term that drives the enemy king to the rim once decisively ahead,
  because a mate is worth more than the whole board
- Our own material weighted at 98% against the opponent's 100% — see below

### A private move generator

The engine this agent plays on checks legality by making every candidate move
and rescanning the whole board. Replacing that inside the agent with a
48-square array, precomputed attack tables and incremental make/unmake gave
**6,200 → 292,000 nodes a second, about 47×**, which is depth 3–4 → depth
10–12.

The engine itself is untouched. The private board is a read-only mirror rebuilt
from the engine's board at the start of every turn, and the move finally
returned is re-validated against the engine's own `get_legal_moves()` before it
leaves the function — so a bug in the generator can't produce an illegal move.

Correctness was established by equivalence testing rather than assumed: the
agent's legal-move set was compared against the engine's at every position
across hundreds of random games, and every piece of incremental state was
checked to return exactly to its prior value after each make/unmake pair. Over
**217,000 position-checks: zero mismatches, zero state corruptions.**

### Opening book

Every position reachable in the first four plies, de-duplicated by board and
side to move — 379 positions — each solved offline at a fixed depth of 12 and
parallelised across cores. Keyed by the same Zobrist hash the agent computes at
runtime, with the move packed into 16 bits, stored as a single hex string and
unpacked into a dict once at import.

It's a cache, not a stronger search: depth 12 is roughly what the live search
reaches in the opening anyway, since branching there is tiny — only 7 legal
first moves. What it buys is time: 0.05 ms instead of about 1.7 s, saving
roughly 3.4 s of the 60-second budget, which matters now that peak usage is
55.6 s.

---

## What failed

### The central idea was wrong

The scoring is non-zero-sum, so the plan was to make the agent trade very
freely: weight our own material well below the opponent's, so an even trade
reads as clearly positive. This is a single percentage, `lw`, applied to our own
material. I set it to 78%. Then measured:

| Our-material weight | Points per game |
| --- | --- |
| 68 | 291 |
| 78 | 481 |
| 92 | 515 |

The greedy setting is catastrophically worse. The reason is the `loss = 0` rule:
bleeding material to bank captures gets you mated, and a mate is worth more than
every piece on the board combined. Sound chess beats point-farming here.

The mechanism survived, at a near-symmetric 98%. The theory behind it did not.

### Four things measured as no better than nothing

- **Aspiration windows** — gained about a ply nominally; 463.3 against 470.0 in
  self-play. Kept, unproven.
- **Spending more of the clock** — the agent was using 27.7s of 60s; doubling
  early thinking time gave 426.7 against 435.0. Kept, unproven.
- **Fine-tuning `lw` inside 86–96** — a 300-game sweep found the whole band
  indistinguishable, best minus worst 24.4 points at 1.2σ. This conclusion later
  turned out to be conditional; see below.
- **The opening book** — can't be measured by the usual harness at all, because
  self-play games start past the book's range.

### Two things rejected before building

- **Select-on-demand move ordering.** Avoiding a full sort when cutoffs happen
  early is a win in a compiled engine. Here it loses: Python's Timsort is
  implemented in C, and an interpreted selection loop over twelve moves is
  slower than just calling `sort()`. Building it would have made the agent
  worse.
- **Staged move generation.** Worth about 10% by the profile, roughly 0.15 ply,
  but it needs the transposition-table move validated against the position or
  move ordering collapses and the tree grows more than the saving. Not worth the
  bug risk for 0.15 ply.

### Two real bugs

**The static exchange evaluation was silently broken.** The fold-back loop
collapsed `-max(a, b)` with the wrong sign. The effect: it discarded 4.68% of
genuinely good captures, worst case throwing away a won rook. It was caught only
by checking SEE against a brute-force recursive swap search instead of trusting
the transcribed formula. After the fix, 0.12% — all of them pin-related cases
SEE structurally cannot see.

**The clock limit was hardcoded.** Tested at a 15-second control, the agent
budgeted as though it had 60 seconds and forfeited every game — an apparent 0–8
defeat that was entirely an artifact of the test harness, not weakness in the
agent. The fix was making the assumed clock a class attribute, which is also
what made every later timed measurement valid.

### Predictions the data contradicted

Worth recording separately, because each one was reasonable and still wrong:

1. *"Non-zero-sum scoring means trade greedily."* The opposite: 291 against 515.
2. *"SEE costs 65% overhead so it will give up half a ply."* Predicted twice, and
   it cost nothing — 12.30 ply with SEE against 12.16 without. A fixed-depth
   benchmark can only show SEE's cost; at fixed time its pruning shrinks the
   quiescence tree by about as much as the calls cost.
3. *"Re-tuning `lw` is wasted, that band is already known to be flat."* The
   optimum had moved once the evaluation changed. The earlier flat result was
   conditional on the old evaluation and didn't generalise.
4. *"Fixed-depth testing systematically overvalues expensive evaluation terms."*
   Named as a pattern after one suspicious six-game result, then disproved: the
   new evaluation costs about twice as much per node and still reaches the same
   depth on the clock, 13.35 against 13.44.

---

## What worked

**The private move generator** — 47× the node rate, depth 3–4 → 10–12. The
single biggest win, described above.

**Rook and king evaluation terms** — the board is six wide and eight deep with
two rooks a side, so files are short and rooks decide games. A bonus for a rook
on a file with none of our own pawns, a small per-square mobility term, and a
phase-scaled king pawn shield: **+37.3 points a game at 2.65σ over 160 games**,
and it more than doubled the checkmate rate, 14 against 6. The only change in
the project to clear statistical significance.

**Re-tuning greed after the evaluation changed** — dismissed in advance as a
waste of time and run anyway. The optimum had shifted and was now monotonic:

| Our-material weight | Points per game |
| --- | --- |
| 86 | 402.2 |
| 92 | 419.8 |
| 98 | 435.0 |

About 2.1σ. One of the more valuable runs, and the earlier 300-game sweep
couldn't have found it, because that sweep measured a different evaluation
function.

**SEE, after the fix** — at the real 60-second control, depth was 12.30 with SEE
against 12.16 without, effectively identical. It buys knowledge for free,
because pruning losing captures shrinks the quiescence tree by about as much as
the SEE calls cost.

**Pin-based legality filtering** — attack detection was the hottest function in
the profile, 16% of runtime across 378,000 calls. But when not in check, a move
can only be illegal if it moves the king or moves a piece pinned against the
king. So the pinned set is computed once per node and everything else skips
verification entirely. Verified for the strong property that it never admits an
illegal move, not merely that it usually agrees: **33,877 positions, zero
admissions.**

**Structural safety** — the clock can't be overrun, because each move's budget
is a fraction of what remains, capped at 15% of the remaining pool. Exhausting
it is arithmetically impossible. Simulated worst case over 75 moves assuming a
pessimistic 10% overshoot on *every single move*: 57.1s of 60s. Measured worst
case in real games: 55.6s, zero forfeits. And if the search ever raises, the
agent replays the last fully-searched root move rather than the first legal one.

---

## Engineering notes

A few things about the methodology that mattered as much as the agent code:

**Measure before optimising.** The first thing built wasn't the agent, it was a
benchmark of the engine. That produced the numbers everything else was planned
around — including that **96% of random-vs-random games hit the 150-turn limit**
(288 of 300). If nearly every game ends in a draw, most matches are decided by
accumulated captures rather than mate, and the evaluation has to weight material
far more heavily than a normal chess engine would.

**The agent is deterministic**, so two self-play games with different random
seeds replay identically. Every match therefore starts from a randomised 4-ply
opening, or a hundred games is really one game measured a hundred times.

**Timed games can't be parallelised**, because wall-clock budgets contend with
each other. So parameter tuning runs at *fixed depth* instead, which is
CPU-load-independent and spreads across all available cores — 300 games in ten
minutes rather than forty games in ninety. Anything that changes the agent's
*speed* then gets its final verdict from real 60-second games, single-process,
because fixed depth can't see a speed cost.

**Every result carries a standard error.** That's what stopped three changes from
shipping as improvements when they were noise, and what identified the one that
was real.

**Read the rules for exploitable structure.** Three things turned out to be worth
real points: no promotion means a pawn on the far rank is a permanently
immobile, dead piece — so the piece-square table penalises it at −34 where a
naive table would reward it; no repetition or fifty-move rule is enforced, so
repeating positions costs nothing; and the engine never validates the move you
hand it, which is why the private generator re-validates its own output.

---

## Running it

```bash
python3 game_runner.py
```

This pits two agents against each other with the 60-second clock and prints the
board, the move log and the running score. Edit the `run_game(...)` call at the
bottom of that file to choose which agents play.

### Layout

| File | What it is |
| --- | --- |
| `minichess_agent.py` | The agent. Single file, no dependencies beyond the standard library. |
| `baseline_agent.py` | The earlier version, kept as the measurement baseline. |
| `board.py`, `config.py` | The game engine. Third party — see `NOTICE`. |
| `game_runner.py` | Match runner. Third party. |
| `p22cs201.py`, `p25cs0004.py` | Sample random-move opponents. Third party. |
| `docs/GameEngine.md` | Engine documentation. Third party. |
| `NOTICE` | Which files the MIT licence does and doesn't cover. |

The self-play harnesses — the fixed-depth gauntlet, the timed match runner and
the book generator — lived in a temporary directory during development and
weren't kept. They can be rebuilt from the descriptions above.

---

## Limitations

- **The 60-second clock is assumed, not observed.** Nothing in the engine
  exposes the real time limit, so it's taken from the runner's default. A
  shorter control would mean a forfeit and a zero. Being defensive enough to
  survive a four-times-wrong assumption would cost about two ply, which was
  judged the worse trade — deliberately, not by oversight.
- **SEE ignores pins.** It assumes every attacker can legally recapture, which
  is the source of the residual 0.12% error rate. This is why SEE is used only
  for pruning inside quiescence and for ordering, never to choose the move.
- **The piece-square tables were never tuned.** Per-piece scale factors are
  exposed as a tunable and the infrastructure exists, but the sweep was never
  run. Every number in those tables is hand-picked, which makes them the largest
  remaining untested surface in the agent.
- **Several constants are estimates.** The stalemate contempt, the repetition
  penalty, the late-move reduction amounts and the null-move reduction were all
  chosen by judgement and never measured.
- **The last round of changes is unproven.** Significant at fixed depth, a tie on
  the clock. See the results table.

---

## Licence

MIT, see `LICENSE`. It covers the agent and this write-up; the game engine and
sample opponents are third party and keep their original terms, which `NOTICE`
sets out.
