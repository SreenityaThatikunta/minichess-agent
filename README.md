# spartans minichess agent

an adversarial search agent for a 6x8 minichess variant, written for the
spartans chess tournament assignment. it reaches about 13 ply in a 60-second
game and scores the maximum possible result against every opponent it was
tested against.

this readme is the full write-up: what the problem actually was, how it was
approached, what failed, what worked, and what the final thing looks like.
everything here was measured rather than assumed, and the places where the
reasoning turned out to be wrong are written down too, because that is most of
what happened.

a note on style: the prose is lowercase throughout. three things keep their
original casing because they have to - the `LICENSE` filename (so github
detects it), the mit licence body (standard legal text, altering it is bad
practice), and `B23CS1072.py` / class `B23CS1072`, which the assignment
mandates exactly.

---

## the problem

### the game

a chess variant on a 6-wide, 8-deep board. twelve pieces a side. all pieces
move as in standard chess, with these removals:

- no pawn promotion
- no castling
- no en passant
- no double pawn push - pawns move one square only

a 60-second clock each side, and a hard limit of 150 total turns (75 a side).
exceeding the clock loses the match outright.

one detail worth knowing: the assignment pdf says each side has 2 knights, 1
bishop and 4 pawns, but the engine actually deals 1 knight, 2 bishops and 6
pawns. the engine is what runs, so the agent is built to the engine.

### the scoring, which is the whole problem

this is the part that makes it not-chess. a match is worth:

| outcome | your score |
| --- | --- |
| checkmate | 600, absolute - it overrides captured material |
| loss | 0, regardless of what you captured |
| draw, stalemate, or turn limit | the value of the pieces you captured |

capture values: pawn 20, bishop 70, knight 70, rook 100.

two consequences fall straight out of this:

1. **your opponent's captures never subtract from your score.** the game is not
   zero-sum. an even trade banks points for you and costs you nothing directly.
2. **total capturable material is 530, which is less than the 600 a mate pays.**
   so mating always beats farming, and losing wipes out everything you earned.

that ordering - never lose, then mate, then capture - drives every design
decision in the agent.

### constraints

- one single python file, named `[rollnumber].py`, class named the same
- 60 kb maximum file size
- the engine must not be modified; graders run their own original copy
- must be adversarial game-tree search, no reinforcement learning

---

## how it was approached

### measure before optimising

the first thing done was not writing an agent, it was benchmarking the provided
engine. that produced the numbers everything else was planned around:

- the engine generates about 100,000 leaf nodes a second
- branching factor is 7 to 20, averaging around 12
- only 7 legal moves exist in the opening position - the back rank is blocked
  and pawns are single-step
- **96% of random-vs-random games hit the 150-turn limit** (288 of 300)

that last number mattered most. if nearly every game ends in a draw at the turn
limit, then most matches are decided by accumulated captures rather than by
mate, and the evaluation needs to care about material far more than a normal
chess engine would.

the starting agent was also measured: depth 3-4, about 6,200 nodes a second, and
it used only 11-30 seconds of its 60-second budget.

### read the rules for exploitable structure

three things in the rules turned out to be worth real points:

- **no promotion** means a pawn that reaches the far rank has zero legal moves
  for the rest of the game. it is a dead piece. the piece-square table shipped
  in `config.py` rewards that square at +30; this agent penalises it at -34.
- **no repetition or fifty-move rule is enforced anywhere** in the engine, so
  repeating positions costs nothing.
- **the engine never validates the move you hand it.** so an agent with its own
  move generator must be certain it is correct, or it will corrupt the game.

### test properly, which was most of the work

the agent is deterministic, so two self-play games with different random seeds
replay identically. every match therefore starts from a randomised 4-ply
opening, or a hundred games is really one game measured a hundred times.

timed games cannot be parallelised, because wall-clock budgets contend with each
other. so parameter tuning runs at **fixed depth** instead, which is
cpu-load-independent and spreads across all available cores - 300 games in ten
minutes rather than forty games in ninety. anything that changes the agent's
*speed* then gets its final verdict from real 60-second games, single-process,
because fixed depth cannot see a speed cost.

and every result is reported with a standard error. that is what stopped three
changes from shipping as improvements when they were noise.

---

## what failed

### the central idea was wrong

the scoring is non-zero-sum, so the plan was to make the agent trade very
freely: weight our own material well below the opponent's, so that an even trade
reads as clearly positive. this is implemented as a single percentage, `lw`, on
our own material.

set to 78%. then measured:

| our-material weight | points per game |
| --- | --- |
| 68 | 291 |
| 78 | 481 |
| 92 | 515 |

the greedy setting is catastrophically worse. the reason is the `loss = 0` rule:
bleeding material to bank captures gets you mated, and a mate is worth more than
every piece on the board combined. sound chess beats point-farming here.

the mechanism survived, at a near-symmetric 98%. the theory behind it did not.

### four things measured as no better than nothing

- **aspiration windows** - gained about a ply nominally, 463.3 against 470.0 in
  self-play. kept, unproven.
- **spending more of the clock** - the agent was using 27.7s of 60s; doubling
  early thinking time gave 426.7 against 435.0. kept, unproven.
- **fine-tuning `lw` inside 86-96** - a 300-game sweep found the whole band
  indistinguishable, best minus worst 24.4 points at 1.2 sigma. this conclusion
  later turned out to be conditional - see below.
- **the opening book** - cannot be measured by the usual harness at all, because
  self-play games start past the book's range.

### two things rejected before building

- **select-on-demand move ordering.** avoiding a full sort when cutoffs happen
  early is a win in a compiled engine. here it loses: python's timsort is
  implemented in c, and an interpreted selection loop over twelve moves is
  slower than just calling `sort()`. building it would have made the agent
  worse.
- **staged move generation.** worth about 10% by the profile, roughly 0.15 ply,
  but it needs the transposition-table move validated against the position or
  move ordering collapses and the tree grows more than the saving. not worth the
  bug risk for 0.15 ply.

### two real bugs

**the static exchange evaluation was silently broken.** the fold-back loop
collapsed `-max(a, b)` with the wrong sign. the effect was that it discarded
4.68% of genuinely good captures, worst case throwing away a won rook. it was
caught only by checking see against a brute-force recursive swap search instead
of trusting the transcribed formula. after the fix, 0.12%, all of them
pin-related cases see structurally cannot see.

**the clock limit was hardcoded.** when tested at a 15-second control the agent
budgeted as though it had 60 seconds and forfeited every game - an apparent 0-8
defeat that was entirely an artifact of the test harness, not weakness in the
agent. the fix was making the assumed clock a class attribute, which is also
what made every later timed measurement valid.

### predictions that the data contradicted

worth recording separately, because each one was reasonable and still wrong:

1. "non-zero-sum scoring means trade greedily" - the opposite, 291 against 515.
2. "see costs 65% overhead so it will give up half a ply" - predicted twice, and
   it cost nothing: 12.30 ply with see against 12.16 without. a fixed-depth
   benchmark can only show see's cost; at fixed time its pruning shrinks the
   quiescence tree by about as much as the calls cost.
3. "re-tuning `lw` is wasted, that band is already known to be flat" - the
   optimum had moved once the evaluation changed. the earlier flat result was
   conditional on the old evaluation and did not generalise.
4. "fixed-depth testing systematically overvalues expensive evaluation terms" -
   named as a pattern after one suspicious six-game result, then disproved: the
   new evaluation costs about twice as much per node and still reaches the same
   depth on the clock, 13.35 against 13.44.

---

## what worked

### a private move generator - the biggest single win

the provided engine checks legality by making every candidate move and
rescanning the whole board. replacing that inside the agent with a 48-square
array, precomputed attack tables and incremental make/unmake gave:

- **6,200 to 292,000 nodes a second, about 47x**
- **depth 3-4 to depth 10-12**

the engine itself is untouched. the private board is a read-only mirror rebuilt
from `engine.board` at the start of every turn, and the move finally returned is
re-validated against the engine's own `get_legal_moves()` before it leaves the
function, so a bug in the generator cannot produce an illegal move.

correctness was established by equivalence testing rather than assumed: the
agent's legal-move set was compared against the engine's at every position
across hundreds of random games, and every piece of incremental state was
checked to return exactly to its prior value after each make/unmake pair. over
**217,000 position-checks, zero mismatches and zero state corruptions.**

### rook and king evaluation terms - the only statistically significant gain

the board is six wide and eight deep with two rooks a side, so files are short
and rooks decide games. added:

- a bonus for a rook on a file with none of our own pawns
- a small per-square rook mobility term
- a phase-scaled king pawn shield

**+37.3 points per game at 2.65 sigma over 160 games**, and it more than doubled
the checkmate rate, 14 against 6. this is the only change in the project to
clear significance.

### re-tuning greed after the evaluation changed

this was dismissed in advance as a waste of time and run anyway. the optimum had
shifted and was now monotonic:

| our-material weight | points per game |
| --- | --- |
| 86 | 402.2 |
| 92 | 419.8 |
| 98 | 435.0 |

about 2.1 sigma. one of the more valuable runs, and the earlier 300-game sweep
could not have found it, because that sweep measured a different evaluation
function.

### static exchange evaluation, after the fix

at the real 60-second control, depth was **12.30 with see against 12.16
without** - effectively identical. it buys knowledge for free, because pruning
losing captures shrinks the quiescence tree by about as much as the see calls
cost. kept.

### pin-based legality filtering

attack detection was the hottest function in the profile, 16% of runtime across
378,000 calls. but when not in check, a move can only be illegal if it moves the
king or moves a piece pinned against the king. so the pinned set is computed
once per node and everything else skips verification entirely.

verified for the strong property that it never admits an illegal move, not
merely that it usually agrees: **33,877 positions, zero admissions.**

### structural safety

- **the clock cannot be overrun.** each move's budget is a fraction of what
  remains, capped at 15% of the remaining pool, so exhausting it is
  arithmetically impossible. simulated worst case over 75 moves assuming a
  pessimistic 10% overshoot on every single move: 57.1s of 60s. measured worst
  case in real games: 55.6s, zero forfeits.
- **the failure path plays a sensible move.** if the search ever raises, the
  agent replays the last fully-searched root move, and failing that picks by
  capture value while checking whether the destination hangs. it used to play
  `legal[0]`, which could be anything.

---

## the final solution

### search

iterative-deepening negamax with alpha-beta, so it can stop on the clock at any
point and still return the best move from the last completed iteration. on top
of that:

- principal-variation search
- transposition table keyed on a 64-bit zobrist hash, 400,000 entries
- killer moves, two per ply, and a history heuristic indexed by from-and-to
  square
- null-move pruning, guarded against zugzwang by requiring non-pawn material and
  never firing while in check
- late-move reductions on quiet moves
- check extensions, bounded so perpetual-check lines cannot recurse forever
- quiescence search over captures with full check evasions, see pruning and
  delta pruning
- aspiration windows at the root

### evaluation

scored from the root player's point of view, in points:

- material at the tournament's own capture values, so the search optimises the
  real objective rather than a proxy
- tapered piece-square tables, middlegame and endgame blended by remaining
  material, maintained incrementally so a leaf evaluation is constant-time
- rook open-file bonus, rook mobility, phase-scaled king pawn shield
- an endgame term that drives the enemy king to the rim once we are decisively
  ahead, because a mate is worth 600
- our own material weighted at 98% against the opponent's 100%

### opening book

every position reachable in the first four plies, de-duplicated by board and
side to move - 379 positions - each solved offline at a fixed depth of 12,
parallelised across cores. keyed by the same zobrist hash the agent computes at
runtime, with the move packed into 16 bits, stored as a single hex string and
unpacked into a dict once at import.

it is a cache, not a stronger search: depth 12 is roughly what the live search
reaches in the opening anyway, since branching there is tiny. what it buys is
time - 0.05 ms instead of about 1.7 s, saving roughly 3.4 s of the 60-second
budget, which matters now that peak usage is 55.6 s.

the book move is only played if it appears in the generated legal move list, so
a hash collision cannot produce an illegal move.

### measured result

| test | result |
| --- | --- |
| vs sample random player | 600 points a game, 4-0, checkmate every game |
| vs sample delayed player | 600 points a game, 4-0, checkmate every game |
| vs the original earlier agent | 600 points a game, 4-0, checkmate every game |
| vs its own previous version | tie, -19.5 +/- 62.4, 0.31 sigma, n=20 |
| depth on the clock | 13.35 ply against 13.44 for the previous version |
| time forfeits | zero, across every game played |

600 is the maximum the rules allow. against its own immediate predecessor it is
a statistical tie - the error bars are too wide to resolve a 30-point difference
at twenty games, which would need roughly 400 games or about thirteen hours. so
this last round of work is best described as **verified safe and unproven
better**. what definitely improved is the foundation underneath it.

---

## running it

```
python3 game_runner.py
```

`game_runner.py` pits two agents against each other with the 60-second clock and
prints the board, the move log and the running score. edit the `run_game(...)`
call at the bottom of that file to choose which agents play.

### repo layout

| file | what it is |
| --- | --- |
| `B23CS1072.py` | the agent - this is the whole submission |
| `OLDAGENT.py` | the earlier version, kept as the measurement baseline |
| `board.py`, `config.py` | the provided engine, unmodified |
| `game_runner.py` | the provided match runner |
| `p22cs201.py`, `p25cs0004.py` | the provided sample players |
| `docs/GameEngine.md` | the provided engine documentation |
| `AI_Assignment1_2026.pdf` | the assignment brief |
| `NOTICE` | which files the mit licence does and does not cover |

the self-play harnesses - the fixed-depth gauntlet, the timed match runner and
the book generator - lived in a temporary directory during development and were
not kept. they can be rebuilt from the descriptions above.

---

## honest limitations

- **the 60-second clock is assumed, not observed.** nothing in `GameEngine`
  exposes the real time limit, so it is taken from the assignment pdf and the
  provided runner, which agree. a shorter control would mean a forfeit and a
  zero. being defensive enough to survive a four-times-wrong assumption would
  cost about two ply, which was judged the worse trade - deliberately, not by
  oversight.
- **see ignores pins.** it assumes every attacker can legally recapture, which
  is the source of the residual 0.12% error rate. this is why see is used only
  for pruning inside quiescence and for ordering, never to choose the move.
- **the piece-square tables were never tuned.** per-piece scale factors are
  exposed as a tunable and the infrastructure exists, but the sweep was never
  run. every number in those tables is hand-picked, which makes them the largest
  remaining untested surface in the agent.
- **several constants are estimates.** the stalemate contempt, the repetition
  penalty, the late-move reduction amounts and the null-move reduction were all
  chosen by judgement and never measured.
- **the last round of changes is unproven.** significant at fixed depth, a tie
  on the clock. see the result table above.

---

## licence

mit, see `LICENSE`. it covers the agent and this write-up; the engine and sample
players were provided as course material and keep their original terms, which
`NOTICE` sets out.
