"""
B23CS1072.py -- agent for the 6x8 "Spartans" mini-chess tournament.

Adversarial game-tree agent:
    iterative deepening negamax + alpha-beta (PVS)
    transposition table (Zobrist), killer moves, history heuristic
    null-move pruning, late-move reductions, check extensions
    quiescence search over captures (with check evasions)
    tournament-aware asymmetric evaluation + tapered piece-square tables
    self-managed clock (never flags)

The underlying chess engine (board.py / config.py) is NOT modified. This file
keeps its own fast internal board purely as a search accelerator; every move it
returns is re-validated against the engine's own legal move list before use.
"""

import time
import random

from board import Move
from config import EMPTY_SQUARE

# --------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------
WIDTH = 6
HEIGHT = 8
NSQ = WIDTH * HEIGHT

EMPTY = 0
WP, WN, WB, WR, WK = 1, 2, 3, 4, 5
BP, BN, BB, BR, BK = 9, 10, 11, 12, 13

CODE = {'wP': WP, 'wN': WN, 'wB': WB, 'wR': WR, 'wK': WK,
        'bP': BP, 'bN': BN, 'bB': BB, 'bR': BR, 'bK': BK,
        EMPTY_SQUARE: EMPTY}

# Capture values, taken straight from the tournament points table:
# pawn 20, bishop 70, knight 70, rook 100. These ARE the score, so the search
# optimises the real objective rather than a proxy.
VAL = [0] * 16
VAL[WP] = VAL[BP] = 20
VAL[WN] = VAL[BN] = 70
VAL[WB] = VAL[BB] = 70
VAL[WR] = VAL[BR] = 100
VAL[WK] = VAL[BK] = 0

# Full capturable material for one side: 6*20 + 70 + 2*70 + 2*100 = 530
SIDE_MATERIAL = 530
TOTAL_MATERIAL = 2 * SIDE_MATERIAL

MATE = 1000000
MATE_BOUND = MATE - 5000
CONTEMPT = 12          # stalemate is mildly bad: it stops us farming points

_SQ = range(NSQ)


def _valid(r, c):
    return 0 <= r < HEIGHT and 0 <= c < WIDTH


def _mk_tables():
    knight, king = [], []
    brays, rrays = [], []
    wpush, bpush, wcap, bcap = [], [], [], []
    watt, batt = [], []
    for sq in _SQ:
        r, c = divmod(sq, WIDTH)

        knight.append(tuple(
            (r + dr) * WIDTH + (c + dc)
            for dr, dc in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                           (1, 2), (1, -2), (-1, 2), (-1, -2))
            if _valid(r + dr, c + dc)))

        king.append(tuple(
            (r + dr) * WIDTH + (c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1))
            if _valid(r + dr, c + dc)))

        def ray(dirs):
            out = []
            for dr, dc in dirs:
                line = []
                rr, cc = r + dr, c + dc
                while _valid(rr, cc):
                    line.append(rr * WIDTH + cc)
                    rr += dr
                    cc += dc
                if line:
                    out.append(tuple(line))
            return tuple(out)

        brays.append(ray(((1, 1), (1, -1), (-1, 1), (-1, -1))))
        rrays.append(ray(((1, 0), (-1, 0), (0, 1), (0, -1))))

        # white marches toward row 0, black toward row HEIGHT-1; single step only
        wpush.append((r - 1) * WIDTH + c if _valid(r - 1, c) else -1)
        bpush.append((r + 1) * WIDTH + c if _valid(r + 1, c) else -1)
        wcap.append(tuple((r - 1) * WIDTH + (c + dc)
                          for dc in (-1, 1) if _valid(r - 1, c + dc)))
        bcap.append(tuple((r + 1) * WIDTH + (c + dc)
                          for dc in (-1, 1) if _valid(r + 1, c + dc)))

        # squares a hostile pawn would have to stand on to attack `sq`
        watt.append(tuple((r + 1) * WIDTH + (c + dc)
                          for dc in (-1, 1) if _valid(r + 1, c + dc)))
        batt.append(tuple((r - 1) * WIDTH + (c + dc)
                          for dc in (-1, 1) if _valid(r - 1, c + dc)))
    return (knight, king, brays, rrays, wpush, bpush, wcap, bcap, watt, batt)


(KNIGHT_T, KING_T, BISHOP_R, ROOK_R,
 WP_PUSH, BP_PUSH, WP_CAP, BP_CAP, WP_ATT, BP_ATT) = _mk_tables()

# --------------------------------------------------------------------------
# Piece-square tables, written for THIS variant (row 0 = black's back rank).
# The decisive quirk: there is no pawn promotion, so a pawn that reaches the
# far rank is a permanently immobile, worthless piece. The stock tables in
# config.py reward exactly that; these punish it.
# --------------------------------------------------------------------------
_PAWN_MG = [
    -34, -34, -34, -34, -34, -34,
     -8,  -6,  -4,  -4,  -6,  -8,
      0,   3,   7,   7,   3,   0,
      3,   7,  13,  13,   7,   3,
      3,   7,  11,  11,   7,   3,
      0,   4,   7,   7,   4,   0,
      0,   0,   2,   2,   0,   0,
      0,   0,   0,   0,   0,   0,
]
_PAWN_EG = [
    -40, -40, -40, -40, -40, -40,
    -14, -12, -10, -10, -12, -14,
     -2,   0,   2,   2,   0,  -2,
      2,   4,   6,   6,   4,   2,
      4,   6,   8,   8,   6,   4,
      2,   5,   7,   7,   5,   2,
      0,   2,   4,   4,   2,   0,
      0,   0,   0,   0,   0,   0,
]
_KNIGHT = [
    -26, -14, -12, -12, -14, -26,
    -12,   0,   5,   5,   0, -12,
     -6,   7,  14,  14,   7,  -6,
     -4,   9,  18,  18,   9,  -4,
     -4,   9,  18,  18,   9,  -4,
     -6,   7,  14,  14,   7,  -6,
    -10,   2,   6,   6,   2, -10,
    -24, -14, -10, -10, -14, -24,
]
_BISHOP = [
    -14,  -7,  -7,  -7,  -7, -14,
     -4,   5,   3,   3,   5,  -4,
     -2,   5,   8,   8,   5,  -2,
     -2,   3,  10,  10,   3,  -2,
     -2,   3,  10,  10,   3,  -2,
     -2,   5,   8,   8,   5,  -2,
     -4,   6,   4,   4,   6,  -4,
    -14,  -8,  -6,  -6,  -8, -14,
]
_ROOK_MG = [
      6,  10,  13,  13,  10,   6,
     14,  18,  20,  20,  18,  14,
      0,   3,   7,   7,   3,   0,
      0,   3,   7,   7,   3,   0,
      0,   3,   7,   7,   3,   0,
      0,   3,   6,   6,   3,   0,
      0,   2,   4,   4,   2,   0,
      0,   3,   7,   7,   3,   0,
]
_ROOK_EG = [
      8,  10,  11,  11,  10,   8,
     10,  12,  13,  13,  12,  10,
      2,   4,   6,   6,   4,   2,
      2,   4,   6,   6,   4,   2,
      2,   4,   6,   6,   4,   2,
      2,   4,   6,   6,   4,   2,
      2,   4,   5,   5,   4,   2,
      0,   2,   4,   4,   2,   0,
]
_KING_MG = [
    -70, -70, -70, -70, -70, -70,
    -56, -56, -56, -56, -56, -56,
    -42, -42, -42, -42, -42, -42,
    -30, -30, -30, -30, -30, -30,
    -20, -22, -24, -24, -22, -20,
    -10, -14, -18, -18, -14, -10,
     -2,  -6, -10, -10,  -6,  -2,
     10,  12,   2,   2,  12,  10,
]
_KING_EG = [
    -34, -18, -10, -10, -18, -34,
    -18,   2,  10,  10,   2, -18,
    -10,  10,  20,  20,  10, -10,
     -8,  14,  24,  24,  14,  -8,
     -8,  14,  24,  24,  14,  -8,
    -10,  10,  20,  20,  10, -10,
    -18,   2,  10,  10,   2, -18,
    -34, -18, -10, -10, -18, -34,
]


def _mirror(sq):
    r, c = divmod(sq, WIDTH)
    return (HEIGHT - 1 - r) * WIDTH + c


def _mk_pst():
    white_mg = {WP: _PAWN_MG, WN: _KNIGHT, WB: _BISHOP,
                WR: _ROOK_MG, WK: _KING_MG}
    white_eg = {WP: _PAWN_EG, WN: _KNIGHT, WB: _BISHOP,
                WR: _ROOK_EG, WK: _KING_EG}
    mg = [[0] * NSQ for _ in range(16)]
    eg = [[0] * NSQ for _ in range(16)]
    for piece, table in white_mg.items():
        for sq in _SQ:
            mg[piece][sq] = table[sq]
            mg[piece + 8][sq] = table[_mirror(sq)]
    for piece, table in white_eg.items():
        for sq in _SQ:
            eg[piece][sq] = table[sq]
            eg[piece + 8][sq] = table[_mirror(sq)]
    return mg, eg


PMG, PEG = _mk_pst()

# Zobrist keys. ZK[0] is all zeros so "no capture" needs no branch.
_rng = random.Random(20260913)
ZK = [[0] * NSQ] + [[_rng.getrandbits(64) for _ in _SQ] for _ in range(15)]
ZSIDE = _rng.getrandbits(64)

# The up-to-three squares directly in front of a king, per colour, used for
# the pawn-shield term.
def _mk_shield():
    w, b = [], []
    for sq in _SQ:
        r, c = divmod(sq, WIDTH)
        w.append(tuple((r - 1) * WIDTH + (c + d)
                       for d in (-1, 0, 1) if _valid(r - 1, c + d)))
        b.append(tuple((r + 1) * WIDTH + (c + d)
                       for d in (-1, 0, 1) if _valid(r + 1, c + d)))
    return w, b


W_SHIELD, B_SHIELD = _mk_shield()
SHIELD = (W_SHIELD, B_SHIELD)

# edge distance of every square, for driving a bare king to the rim
EDGE = [min(sq // WIDTH, HEIGHT - 1 - sq // WIDTH) +
        min(sq % WIDTH, WIDTH - 1 - sq % WIDTH) for sq in _SQ]


# Opening book: {zobrist hash: packed move} for every position reachable in
# the first 4 plies, solved offline at a fixed depth 12 -- the same depth the
# live search reaches here, since opening branching is tiny. So this is a cache,
# not a stronger search: it returns in 0.05ms instead of ~1.7s, saving roughly
# 3.4s of the 60s budget across the two book moves each side gets.
_BOOK_DATA = (
    "003d66f86ecfc2d008260064b40afbd300f907ea00a28ad3cc441a1b0826013444e15d35"
    "13d20340016d9613c829d1fb034001aba8caffbecb19040a023d6cb3f9c7d6ac049f0264"
    "be416cdb1485092b02c352c99389a3ce368d03344eaaca3d07ae0388036d9c585f21c587"
    "034003f7666c8f172fc90388045f0a38ef0fa484040a04c0e6134d847c4f068d052dd616"
    "63cf27530340058e22d579a9642c0340065f00737807b0f8078506993eaa4f90aa1a068d"
    "072ddc5df4c7332f050507901cb37c6a7b1808260835d1dc0de3311e038808caf7563c36"
    "da2a0340096b4a919d79b691034009c3d54f0fcc0b2807ea09f1b0a54d4f5cdf040a0a93"
    "2fef3e220c7f03400b6b40da0a71a2ed03400bd8dde42e0df0fa040a0c59d6baba5fc3ee"
    "07850c92dedb3d5075d833070c9fe8638dc8d90c060d0d2405b8053f63ff05050d96ca7a"
    "be32080e07ea0d9ee53106358d4503400e59dcf12d57d79203400e796673b88fd0b80388"
    "0ec326c5fd613ddc040a0ec630da8fdc0f5903880f240ff39237778303400f9eef7a913d"
    "9939034010e67391eb11a982034011d62e3d08ca63f5034714212e166fcaaa1c03881644"
    "4197ec86e7eb03c11762d01b3fc643ad040a17d4e5f15b3fc881040a18b33cc03d766795"
    "038818cfb03966b2fac0054f19f7f166d618ab7503401acfba72f1baeebc16451ae0a558"
    "2949da9403401c75b5be9d7e1bd703401cc737e3c45bdeaa250d1d944e7b1f8321e60388"
    "1e75bff50a760fab03401f634c0f6274d543040a20354bf1f9a242a6018121e3bd930361"
    "4e5703402280a4a4666f88bc03402314bfe77e96baf2040a263aa1239da369ab038828c4"
    "b00b8d937cff040a29fc4e083d0f429f34cc2d2dc4863f1b31d1040a2f2dcecda81325ad"
    "05902f56dfa169dff8e103883046c146eec9944d0340305f572eb63af76d07ea31567537"
    "85c0266f034031beae73da9a3adf03c532122f475c45eecc3307321f19ffecdd4218040a"
    "330fad8e87d4f03a038833163be6df27931a07ea33245e0c9da4c4ed034033b1a0b20006"
    "2ae7038833bea4384d922ea3040a34071e059cf4a90f08a83413de735f37976b03883416"
    "c86c2d8aa5ee040a348c3258fdbc4fa0034034e7eb03851986f50388350e3c1caf0e780d"
    "038835582c784f186bee040a356b1b6e92ea05ff0340364a06ca5d23413e060d368c3813"
    "6ab45bdc078536b520406cf6aa0a0340374324d36ed9903c07ea3757e4a5ad1aae580388"
    "37c2384be963ca48330737d1f1ac06e8ff3903403819cf362e85310e0340382baadc6c06"
    "66f907ea3910ed2f1d7fe00c07ea392288c55ffcb7fb034039b872f18fca5db503403a2f"
    "00d0fd05cf1e03883a40178f2c91e75b040a3a7272656e12b0ac051f3b24b12d9499c72b"
    "01813b7b507c5de861ae03883bb878ba18c249c903403c43a355f29c38da040a3dc13c03"
    "ee7c022c05053e101ea5efd2d6f8040a3e227b4fad51810f08263f2b59569eab500d03c1"
    "3f62c28b6312552f03403fc136487974165003404041fa780ddba621040a406f1fe56516"
    "a073040a4087c4a13a4cbcc3034040de1653af507eea040a41332656811b25f6040a418e"
    "e6b809b66dc108264287ceeaad44a8bf038842de1c1838586a9603404326732d0c0bc404"
    "040a438eecf39ebe79bd08a8442396b88c1bc075051f443756ce4fd8fe11038844bc7a93"
    "2e9018be082644dda8c2e655b5170340452ab4a1bfe11177068d455f3794fab58fe1040a"
    "45b5588a1d6ac9bc054f467a4e018e0f1620092b46bc70d8b9980cc2082646dda289715d"
    "a16b040a47736c18bdf5c722060d47b552c18a62ddc01785481be217bf2a31e703404881"
    "18236f1cdba9040a48d8cad1fa00198003884912c00e8cd0e0e508a84920a5e4ce53b712"
    "040a49801571e4e18fe0040a49883a3a5ce60aab086f4a423aaebd3ee7b203884a472cb1"
    "cf83d537040a4a811268f814cfd503404ad8c09a6d080dfc03404b3affb370e37526040a"
    "4b883071cbee1ed707ea4c2b219cef3733bb03884c43cb9abe4c9e1533074d75bad17fad"
    "b43403884dc627ef5bd1e62303884dfe53378b75fe2803884e1233847e7dd611040a4e2b"
    "2bd7783f27c703884fc62da4ccd9f25f040a5005033a2a8e15b807ea50e5f63c33633a42"
    "018151073b7c8bd68838040a510c21231974c4ba03405355f99a1b6012ef0388569554d6"
    "6f9a0c26038857b3bd8db67c9650038857b3c55abcdaa860050e586b45fe7faa1972060d"
    "597db60417a8c39a0340598abe3bfd572343054f5b7dbc4f80a0d7e6040a5b8ab4706a5f"
    "373f03885cd140798466f865040a5e2d1def4b7f73a903885f26ac1222e37b9c040a6132"
    "a8d2807da59a03406251b1e5e5736371050e63c5aaa6fd8a513f034066ebb4621ebf8266"
    "040a693334c6ddcf3374068d694825aa1c03ee3803c96a7b7b4c0ef56c0f040a6b482fe1"
    "8b0bfa44040a6cda404b6f477e5a040a6eda4a00f84f6a260340700d2e33bde395ce040a"
    "70233329d9b016ab03407054fcc128ff57e703407092c2181f684d05040a7097d4076dd5"
    "7f80040a70bc278577a54b570340715dded81b0586e508a8720d24782aeb81b203887254"
    "f68abff7439b034072ce0cbe6fc1a9d50340735dd4938c0d929907ea73c72ea75c3b78d7"
    "07ea73d6f8ceed457436038873f54b4d1eb82f200340740cd54c2999f815330774c7dd2d"
    "ae964e23040a74e46eae5d6b1535038875893939cc048023038875ba0e2f11f6ee320340"
    "760cdf07be91ec69330776643501efea41c703887700e4ed85f414f40340778eb1e5c78b"
    "2a54040a77a05478af462c06030678c52adc7221621d038878c8da77ad99dac3034078fa"
    "bf9def1a8d34082679c1f86e9e630bc107ea79f39d84dce05c3603c17a9102ceaf8d0c96"
    "03887aa36724ed0e5b6108a87abaf14cb5fd384103407baa453ddef48a6303407bdfc608"
    "9ba014f5040a7c92b6147180d31703407cb121ff8b0cdb2203887da682b16d1e5f100388"
    "7df5ec59606217c204cd7ec10be46cce3d35040a7ee2b8679f33662303887ef36e0e2e4d"
    "6ac208267fb3d7cae00ebee2040a7ffa4c171db7bbc0038880a474f12ac7d1d5034081ad"
    "56e8193d00d7082683cc8ef246f0b902040a84ab9c8a20f546f3040a85a02d7749694ec6"
    "040a85cf3a2898fd66830388869fc088a91361d407ea8796e2919ae9b0d60340879b123a"
    "4551080803c9893a0de7d89a2bd201818ad701dffb74ea36040a8c3e8d15b46aab3f040a"
    "8c3ef5c2becc950f03888c9916416b4b12c207ea8d90345858b1c3c003408f7d0bcfeec0"
    "7cbe03408fc9ece15aa51595038891e9afaa3e68b34e040a92ad957ccc458a7d0388941d"
    "0d582ed9dcd303409444dfaabbc51efa034095142f411d230dd108a8952543aa3985061a"
    "0388959f031c7c6beb7e040a95f4da4704ce222b0340961d0713b9d1c8af03409644d5e1"
    "2ccd0a860340967363af8963c34734ce96db39ca8e46d24d040a9714250a8a2b19ad07ea"
    "97f4d00c93c63657050598ab43b50e1df96b04cd98f82d5d0361b1b9068d99bce0fbe80f"
    "7d5903889bef7963fc30c058040a9c9b9240cafa3b2a03409e6c9034b70dcf8f03889ed2"
    "070cf8a3b28e03889fba66564dcec37e040aa02775a5dce2b70c0388a12e57bcef18660e"
    "08a8a13a97ca2cdb586a0388a27ead1cdef6615904cda3778f05ed0cb05b08a8a43ee752"
    "e218838d0388a54c3b7c6ed8005a040aa56f88ff9d255b4c040aa626e437f4383f3b0340"
    "a678dcc162742aad0340a73c1167891ae64d040aa85bc856ef534959040aa8787bd51cae"
    "124f0388a8f39233e87658530388a9cb6c3058ea66330388ac0bc17c2c1078fa0388ac8d"
    "d47587e2299b0388ad6a5d7cae50601a0340ae373eb713e0d35d0388ae5219c52e04aeaf"
    "0340ae59039abca6e22d0388af4aedb5ac80734c040aaf5b3bdc1dfe7fad07eaaf695e36"
    "5f7d285a0340af6a573739587466040aafbbceda041350570388b2be7c89d57f302b0388"
    "b51c02488a4e8da70388b5321f52ee1d0ec20340b64cf8e8bba08af007eab73215197915"
    "1abe0340b745daf1885a5bf20340b9dc817de7655e1e0388ba2b6d1641e0dde7040abb20"
    "dceb287cd5d20388bd6684fa1ca9bf090388bdd406a7458c7a74250fbfd40cecd2846e08"
    "2505c03cfa6d54623f3a0181c14e2643d8a2bced040ac425689a6a980bbb0181c47a89cb"
    "a3e9ad3e040ac52c4a835962dab9051fc5713836ca75a50b040ac67cb023688cddee0340"
    "c74a077bc64de3c50581c775923a5b760cec07eac88fc94a3198952c0340ca352388a59a"
    "6fea0340ca5b1ea50adc5e9a248dcb48f08a1afacffb040acc7a66eaaad4aef80340ccef"
    "9854377640f20388cdc9710fee90da840388cfac1e8e6ddc97730340d003fd18cc0dde76"
    "040ad0c5c3c1fb9ac4940785d10adf01fff70f7408a8d11e1f773c3431100388d1ea2a07"
    "e61a208e040ad24ee5d70dda36470826d25a25a1ce1908230388d2c5c98a6c92d0e80340"
    "d347c7ce3e20e745040ad5f456ebba99edd7040ad5f75feadcbcb1eb0340d6a276ee0a7f"
    "288a0388d71fee433f4648b70388d7f755a14bb4a5970505d96499c5aad303be0340dce6"
    "dd1de1b5b31c0340dda210bb0adb7ffc040addde9c42511fe2a90388ddf1836889ecd681"
    "0340dee6d75676bda7600340df6b7317ced228b30340dfa21af09dd36b800340dff18923"
    "1ee4c2fd0340e1cd27172e87da340388e29dddb71f69dd6308a8e394ffae2c930c610388"
    "e4a669ce9cbd6d620388e524678ace0f5acf040ae5af4bd7af47bc6008a8e5be9dbe1e39"
    "b0810388e6a9c980e168c1600340e7246dc159074eb30285e7ce93cdf08a05b5040ae7ed"
    "04260a060d800340e83ce8fd08b629b80340e88add176c4fa2940388e89b0b7edd31ae75"
    "032fe9922967eecb7f770388eb74a777db8c76c3040aebcbf1deecdfa9220388ecdad43d"
    "af0c9337040aeced62730aa25af60388ed3810d86ee097650388ee830c84ad1845620340"
    "eed51ce04d0e5681040aeeed68389daa4e8a0340ef6adb9b870fbb9a0340ef8a2e9d9ee2"
    "946007eaefb84b77dc61c3970340f12cef120cc90c67068df19d1e23ca1157c90181f4a0"
    "8682a0a326f20388f6a08cc937ab328e0388f6af88437a3f36ca0181f8998f7f7c80879d"
    "040af9caf6e7a75878d103c9fa6e8d0b017773380388fad01a334ed90e390181fc35971d"
    "343ccb3e0388fca95e8ab86745dc0388fd3e26e05da0c30b35d0fe359d56a334df42040a"
    "ff3e2cabcaa8d777040a"
)

# {position hash: packed move}, unpacked once at import.
_BOOK = {int(_BOOK_DATA[i:i + 16], 16): int(_BOOK_DATA[i + 16:i + 20], 16)
         for i in range(0, len(_BOOK_DATA), 20)}


class _TimeUp(Exception):
    pass


class B23CS1072:
    """Tournament agent. Interface: get_best_move(), evaluate_board()."""

    # Weight (percent) on our OWN material in the evaluation. Below 100 it
    # makes even trades profitable, which is correct here because only the
    # material WE capture scores. Raised when fragile, lowered when winning.
    LW_MID = 98
    LW_SAFE = 100
    LW_WIN = 90

    # The tournament control is a 60-second game with at most 75 moves a side
    # (assignment PDF; game_runner.py defaults to the same). The agent paces
    # itself against this and tracks its own spend, so it cannot flag.
    TOTAL_TIME = 60.0
    MAX_MOVES = 75
    USE_ASPIRATION = True
    MOVE_HORIZON = 30
    FIXED_DEPTH = 0        # testing only: >0 searches to a fixed depth

    # Evaluation term weights (0 disables the term). Rooks dominate on a
    # 6-wide board, so file control and rook activity are the terms most
    # likely to matter here.
    EV_ROOK_OPEN = 14      # rook on a file with none of our own pawns
    EV_ROOK_MOB = 2        # per square of rook mobility
    EV_KING_SHIELD = 8     # per friendly pawn directly shielding our king
    USE_SEE = True         # prune/deprioritise losing captures

    # Per-piece multipliers (percent) on the piece-square tables, in the order
    # pawn, knight, bishop, rook, king. Exposed so the tables can be tuned by
    # self-play rather than left at hand-picked values.
    PST_SCALE = (100, 100, 100, 100, 100)

    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 1

        # clock self-management (tournament gives 60s / 75 moves per side)
        self.clock = self.TOTAL_TIME
        self.reserve = self.TOTAL_TIME * 0.035
        self.moves_made = 0

        # search state
        self.tt = {}
        self.killers = [0] * 128
        self.hist = [0] * (NSQ * NSQ)
        self.seen = set()
        self.deadline = 0.0
        self.nodes = 0
        self.last_error = None
        self._fallback_move = None

        # internal board
        self.bd = [EMPTY] * NSQ
        self.occ = (set(), set())
        self.stm = 0
        self.me = 0
        self.ksq = [0, 0]
        self.mat = [0, 0]
        self.npawn = [0, 0]
        self.pfile = ([0] * WIDTH, [0] * WIDTH)
        self.rooks = (set(), set())
        self.pmg = [0, 0]
        self.peg = [0, 0]
        self.hash = 0
        self.lw = 78          # weight on our OWN material (see _evaluate)
        if self.PST_SCALE == (100, 100, 100, 100, 100):
            self._pmg, self._peg = PMG, PEG
        else:
            self._pmg = [list(t) for t in PMG]
            self._peg = [list(t) for t in PEG]
            for kind, pct in enumerate(self.PST_SCALE, start=1):
                for piece in (kind, kind + 8):
                    self._pmg[piece] = [(v * pct) // 100 for v in PMG[piece]]
                    self._peg[piece] = [(v * pct) // 100 for v in PEG[piece]]

    # ------------------------------------------------------------------
    # engine <-> internal board
    # ------------------------------------------------------------------
    def _sync(self):
        board = self.engine.board
        PMG = self._pmg
        PEG = self._peg
        bd = [EMPTY] * NSQ
        white, black = set(), set()
        mat = [0, 0]
        npawn = [0, 0]
        pfile = ([0] * WIDTH, [0] * WIDTH)
        rooks = (set(), set())
        pmg = [0, 0]
        peg = [0, 0]
        ksq = [0, 0]
        h = 0
        for r in range(HEIGHT):
            row = board[r]
            base = r * WIDTH
            for c in range(WIDTH):
                piece = CODE.get(row[c], EMPTY)
                if not piece:
                    continue
                sq = base + c
                bd[sq] = piece
                h ^= ZK[piece][sq]
                side = piece >> 3
                (white if side == 0 else black).add(sq)
                mat[side] += VAL[piece]
                kind = piece & 7
                if kind == 1:
                    npawn[side] += 1
                    pfile[side][c] += 1
                elif kind == 4:
                    rooks[side].add(sq)
                pmg[side] += PMG[piece][sq]
                peg[side] += PEG[piece][sq]
                if piece == WK:
                    ksq[0] = sq
                elif piece == BK:
                    ksq[1] = sq
        self.bd = bd
        self.occ = (white, black)
        self.mat = mat
        self.npawn = npawn
        self.pfile = pfile
        self.rooks = rooks
        self.pmg = pmg
        self.peg = peg
        self.ksq = ksq
        self.stm = 0 if self.engine.white_to_move else 1
        if self.stm:
            h ^= ZSIDE
        self.hash = h

    def _to_move(self, mv):
        frm = mv & 63
        to = (mv >> 6) & 63
        return Move((frm // WIDTH, frm % WIDTH),
                    (to // WIDTH, to % WIDTH), self.engine.board)

    # ------------------------------------------------------------------
    # make / unmake (fully reversible from the packed move alone)
    # ------------------------------------------------------------------
    def _make(self, mv):
        bd = self.bd
        PMG = self._pmg
        PEG = self._peg
        frm = mv & 63
        to = (mv >> 6) & 63
        cap = (mv >> 12) & 15
        piece = bd[frm]
        bd[frm] = EMPTY
        bd[to] = piece
        side = piece >> 3
        occ = self.occ[side]
        occ.discard(frm)
        occ.add(to)
        self.pmg[side] += PMG[piece][to] - PMG[piece][frm]
        self.peg[side] += PEG[piece][to] - PEG[piece][frm]
        kind = piece & 7
        if kind == 5:
            self.ksq[side] = to
        elif kind == 4:
            r = self.rooks[side]
            r.discard(frm)
            r.add(to)
        elif kind == 1:
            fc, tc = frm % WIDTH, to % WIDTH
            if fc != tc:                     # pawns change file only by capturing
                pf = self.pfile[side]
                pf[fc] -= 1
                pf[tc] += 1
        if cap:
            other = cap >> 3
            self.occ[other].discard(to)
            self.mat[other] -= VAL[cap]
            ck = cap & 7
            if ck == 1:
                self.npawn[other] -= 1
                self.pfile[other][to % WIDTH] -= 1
            elif ck == 4:
                self.rooks[other].discard(to)
            self.pmg[other] -= PMG[cap][to]
            self.peg[other] -= PEG[cap][to]
        self.hash ^= (ZK[piece][frm] ^ ZK[piece][to] ^ ZK[cap][to] ^ ZSIDE)
        self.stm ^= 1

    def _undo(self, mv):
        bd = self.bd
        PMG = self._pmg
        PEG = self._peg
        frm = mv & 63
        to = (mv >> 6) & 63
        cap = (mv >> 12) & 15
        piece = bd[to]
        bd[frm] = piece
        bd[to] = cap
        side = piece >> 3
        occ = self.occ[side]
        occ.discard(to)
        occ.add(frm)
        self.pmg[side] += PMG[piece][frm] - PMG[piece][to]
        self.peg[side] += PEG[piece][frm] - PEG[piece][to]
        kind = piece & 7
        if kind == 5:
            self.ksq[side] = frm
        elif kind == 4:
            r = self.rooks[side]
            r.discard(to)
            r.add(frm)
        elif kind == 1:
            fc, tc = frm % WIDTH, to % WIDTH
            if fc != tc:
                pf = self.pfile[side]
                pf[tc] -= 1
                pf[fc] += 1
        if cap:
            other = cap >> 3
            self.occ[other].add(to)
            self.mat[other] += VAL[cap]
            ck = cap & 7
            if ck == 1:
                self.npawn[other] += 1
                self.pfile[other][to % WIDTH] += 1
            elif ck == 4:
                self.rooks[other].add(to)
            self.pmg[other] += PMG[cap][to]
            self.peg[other] += PEG[cap][to]
        self.hash ^= (ZK[piece][frm] ^ ZK[piece][to] ^ ZK[cap][to] ^ ZSIDE)
        self.stm ^= 1

    # ------------------------------------------------------------------
    # attack detection
    # ------------------------------------------------------------------
    def _attacked(self, sq, by_black):
        bd = self.bd
        if by_black:
            for s in BP_ATT[sq]:
                if bd[s] == BP:
                    return True
            knight, bishop, rook, king = BN, BB, BR, BK
        else:
            for s in WP_ATT[sq]:
                if bd[s] == WP:
                    return True
            knight, bishop, rook, king = WN, WB, WR, WK
        for s in KNIGHT_T[sq]:
            if bd[s] == knight:
                return True
        for s in KING_T[sq]:
            if bd[s] == king:
                return True
        for line in BISHOP_R[sq]:
            for s in line:
                p = bd[s]
                if p:
                    if p == bishop:
                        return True
                    break
        for line in ROOK_R[sq]:
            for s in line:
                p = bd[s]
                if p:
                    if p == rook:
                        return True
                    break
        return False

    def _pins(self, side):
        """Squares holding a piece that is pinned against its own king.

        When we are not in check, any move by a non-king, non-pinned piece is
        legal by construction -- so the expensive `_attacked` verification can
        be skipped for it entirely.
        """
        k = self.ksq[side]
        bd = self.bd
        pinned = None
        if side == 0:
            ebishop, erook = BB, BR
        else:
            ebishop, erook = WB, WR
        for rays, slider in ((BISHOP_R[k], ebishop), (ROOK_R[k], erook)):
            for line in rays:
                cand = -1
                for t in line:
                    q = bd[t]
                    if not q:
                        continue
                    if (q >> 3) == side:
                        if cand >= 0:
                            break           # two friendly blockers: no pin
                        cand = t
                    else:
                        if cand >= 0 and q == slider:
                            if pinned is None:
                                pinned = set()
                            pinned.add(cand)
                        break
        return pinned

    def _in_check(self, side):
        return self._attacked(self.ksq[side], side == 0)

    # ------------------------------------------------------------------
    # move generation (pseudo-legal; legality is filtered during search)
    # mirrors board.py exactly: single-step pawns, no promotion, no castling,
    # no en passant.
    # ------------------------------------------------------------------
    def _gen(self, caps_only=False):
        bd = self.bd
        moves = []
        add = moves.append
        white = self.stm == 0
        quiet = not caps_only
        for frm in self.occ[self.stm]:
            piece = bd[frm]
            kind = piece & 7

            if kind == 1:                                   # pawn
                if white:
                    push, caps = WP_PUSH[frm], WP_CAP[frm]
                else:
                    push, caps = BP_PUSH[frm], BP_CAP[frm]
                if quiet and push >= 0 and not bd[push]:
                    add(frm | (push << 6))
                for s in caps:
                    q = bd[s]
                    if q and (q >= 9) != (not white):
                        add(frm | (s << 6) | (q << 12))
                continue

            if kind == 2 or kind == 5:                      # knight / king
                targets = KNIGHT_T[frm] if kind == 2 else KING_T[frm]
                for s in targets:
                    q = bd[s]
                    if not q:
                        if quiet:
                            add(frm | (s << 6))
                    elif (q >= 9) != (not white):
                        add(frm | (s << 6) | (q << 12))
                continue

            rays = BISHOP_R[frm] if kind == 3 else ROOK_R[frm]
            for line in rays:
                for s in line:
                    q = bd[s]
                    if not q:
                        if quiet:
                            add(frm | (s << 6))
                        continue
                    if (q >= 9) != (not white):
                        add(frm | (s << 6) | (q << 12))
                    break
        return moves

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def _evaluate(self):
        """Score from the ROOT player's point of view, in points.

        Tournament scoring is not zero-sum: a match is worth 600 for
        checkmate, 0 for a loss, and otherwise exactly the value of the
        pieces WE captured -- the opponent's captures never subtract from
        us. So the real objective is to minimise the opponent's surviving
        material; our own material only matters instrumentally (it keeps us
        alive and lets us keep capturing). Hence our material carries weight
        `lw` < 100%, which makes even trades correctly profitable while
        still refusing genuine sacrifices.
        """
        mat = self.mat
        me = self.me
        opp = me ^ 1
        total = mat[0] + mat[1]
        phase = (total << 8) // TOTAL_MATERIAL
        if phase > 256:
            phase = 256
        anti = 256 - phase
        pst_me = (self.pmg[me] * phase + self.peg[me] * anti) >> 8
        pst_op = (self.pmg[opp] * phase + self.peg[opp] * anti) >> 8

        score = (mat[me] * self.lw) // 100 + pst_me - mat[opp] - pst_op

        # Positional terms the piece-square tables cannot express. On a
        # 6-wide board the rooks decide most games, so file control and rook
        # activity carry the most weight; the king shield only matters while
        # there is still material to attack with, so it is phase-scaled.
        if self.EV_ROOK_OPEN or self.EV_ROOK_MOB:
            score += self._activity(me) - self._activity(opp)
        if self.EV_KING_SHIELD and phase > 96:
            score += ((self._shield(me) - self._shield(opp))
                      * self.EV_KING_SHIELD * phase) >> 8

        # conversion: with a decisive material edge, hunt the mate (600 pts)
        if mat[me] - mat[opp] >= 100 and mat[opp] <= 140:
            ok = self.ksq[opp]
            mk = self.ksq[me]
            score += (5 - EDGE[ok]) * 9
            dr = abs(ok // WIDTH - mk // WIDTH)
            dc = abs(ok % WIDTH - mk % WIDTH)
            score += (7 - (dr if dr > dc else dc)) * 5
        return score

    def _activity(self, side):
        """Rook file control and mobility for one side."""
        bd = self.bd
        pf = self.pfile[side]
        openb = self.EV_ROOK_OPEN
        mobw = self.EV_ROOK_MOB
        total = 0
        for sq in self.rooks[side]:
            if not pf[sq % WIDTH]:
                total += openb
            if mobw:
                free = 0
                for line in ROOK_R[sq]:
                    for t in line:
                        if bd[t]:
                            break
                        free += 1
                total += free * mobw
        return total

    def _shield(self, side):
        """Friendly pawns on the squares directly in front of our king."""
        bd = self.bd
        pawn = WP if side == 0 else BP
        n = 0
        for t in SHIELD[side][self.ksq[side]]:
            if bd[t] == pawn:
                n += 1
        return n

    def _eval_stm(self):
        s = self._evaluate()
        return s if self.stm == self.me else -s

    def evaluate_board(self, game_state=None):
        """Heuristic score of the live engine position, White's perspective."""
        if game_state == "checkmate":
            return -MATE if self.engine.white_to_move else MATE
        if game_state == "stalemate":
            return 0
        self._sync()
        saved_me, saved_lw = self.me, self.lw
        self.me = 0
        self.lw = 100
        try:
            return self._evaluate()
        finally:
            self.me, self.lw = saved_me, saved_lw

    # ------------------------------------------------------------------
    # static exchange evaluation
    # ------------------------------------------------------------------
    SEE_KING = 10000

    def _least_attacker(self, sq, side, removed):
        """Cheapest piece of `side` attacking `sq`, seeing through `removed`
        (squares vacated earlier in the exchange, which exposes x-rays)."""
        bd = self.bd
        if side == 0:
            for t in WP_ATT[sq]:
                if bd[t] == WP and t not in removed:
                    return t
            knight, bishop, rook, king = WN, WB, WR, WK
        else:
            for t in BP_ATT[sq]:
                if bd[t] == BP and t not in removed:
                    return t
            knight, bishop, rook, king = BN, BB, BR, BK
        for t in KNIGHT_T[sq]:
            if bd[t] == knight and t not in removed:
                return t
        for line in BISHOP_R[sq]:
            for t in line:
                if t in removed:
                    continue
                q = bd[t]
                if q:
                    if q == bishop:
                        return t
                    break
        for line in ROOK_R[sq]:
            for t in line:
                if t in removed:
                    continue
                q = bd[t]
                if q:
                    if q == rook:
                        return t
                    break
        for t in KING_T[sq]:
            if bd[t] == king and t not in removed:
                return t
        return None

    def _see(self, mv):
        """Material the side to move nets from the capture sequence on the
        destination square, assuming both sides keep recapturing."""
        bd = self.bd
        frm = mv & 63
        to = (mv >> 6) & 63
        gain = [VAL[(mv >> 12) & 15]]
        piece = bd[frm]
        aval = self.SEE_KING if piece & 7 == 5 else VAL[piece]
        removed = {frm}
        side = (piece >> 3) ^ 1
        d = 0
        while True:
            att = self._least_attacker(to, side, removed)
            if att is None:
                break
            d += 1
            gain.append(aval - gain[d - 1])
            if -gain[d - 1] < 0 and gain[d] < 0:
                break
            ap = bd[att]
            aval = self.SEE_KING if ap & 7 == 5 else VAL[ap]
            removed.add(att)
            side ^= 1
        while d > 0:
            # gain[d-1] = -max(-gain[d-1], gain[d]); the side to move may
            # always decline to continue the exchange.
            if -gain[d - 1] <= gain[d]:
                gain[d - 1] = -gain[d]
            d -= 1
        return gain[0]

    # ------------------------------------------------------------------
    # move ordering
    # ------------------------------------------------------------------
    def _order(self, moves, ttmove, ply):
        bd = self.bd
        k1 = self.killers[ply << 1]
        k2 = self.killers[(ply << 1) + 1]
        hist = self.hist
        see = self.USE_SEE
        scored = []
        for mv in moves:
            if mv == ttmove:
                s = 1 << 30
            else:
                cap = (mv >> 12) & 15
                if cap:
                    av = VAL[bd[mv & 63]]
                    if see and VAL[cap] < av and self._see(mv) < 0:
                        s = 700000 + (VAL[cap] << 4) - av   # losing capture
                    else:
                        s = 1000000 + (VAL[cap] << 4) - av
                elif mv == k1:
                    s = 900000
                elif mv == k2:
                    s = 880000
                else:
                    s = hist[(mv & 63) * NSQ + ((mv >> 6) & 63)]
            scored.append((s, mv))
        scored.sort(reverse=True)
        return [mv for _, mv in scored]

    # ------------------------------------------------------------------
    # quiescence
    # ------------------------------------------------------------------
    def _quiesce(self, alpha, beta, ply):
        self.nodes += 1
        if not (self.nodes & 1023) and time.perf_counter() > self.deadline:
            raise _TimeUp

        us = self.stm
        in_check = self._in_check(us)
        if in_check:
            if ply > 40:
                return self._eval_stm()
            moves = self._gen(False)
            best = -MATE
        else:
            stand = self._eval_stm()
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            if ply > 24:
                return stand
            moves = self._gen(True)
            best = stand

        bd = self.bd
        scored = []
        for mv in moves:
            cap = (mv >> 12) & 15
            scored.append(((VAL[cap] << 4) - VAL[bd[mv & 63]] if cap else -1, mv))
        scored.sort(reverse=True)

        legal = 0
        use_see = self.USE_SEE
        for _, mv in scored:
            cap = (mv >> 12) & 15
            if not in_check:
                if best + VAL[cap] + 90 < alpha:
                    continue
                # A capture that loses material on the swap-off cannot help
                # here; searching it only inflates the quiescence tree.
                if use_see and VAL[cap] < VAL[self.bd[mv & 63]] \
                        and self._see(mv) < 0:
                    continue
            self._make(mv)
            if self._attacked(self.ksq[us], us == 0):
                self._undo(mv)
                continue
            legal += 1
            score = -self._quiesce(-beta, -alpha, ply + 1)
            self._undo(mv)
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        return best
        if in_check and legal == 0:
            return -MATE + ply
        return best

    # ------------------------------------------------------------------
    # main search
    # ------------------------------------------------------------------
    def _search(self, depth, alpha, beta, ply, allow_null):
        self.nodes += 1
        if not (self.nodes & 1023) and time.perf_counter() > self.deadline:
            raise _TimeUp

        alpha_orig = alpha
        key = self.hash
        entry = self.tt.get(key)
        ttmove = 0
        if entry is not None:
            e_depth, e_flag, e_score, e_move = entry
            ttmove = e_move
            if e_depth >= depth:
                if e_score > MATE_BOUND:
                    e_score -= ply
                elif e_score < -MATE_BOUND:
                    e_score += ply
                if e_flag == 0:
                    return e_score
                if e_flag == 1:
                    if e_score >= beta:
                        return e_score
                elif e_score <= alpha:
                    return e_score

        us = self.stm
        in_check = self._attacked(self.ksq[us], us == 0)
        if in_check and ply < 32:
            depth += 1                                   # check extension
        if depth <= 0:
            return self._quiesce(alpha, beta, ply)

        # null-move pruning (skipped in check and in pawn-only endings)
        if (allow_null and not in_check and depth >= 3
                and beta < MATE_BOUND
                and self.mat[us] > 20 * self.npawn[us]):
            self.stm ^= 1
            self.hash ^= ZSIDE
            try:
                score = -self._search(depth - 3, -beta, -beta + 1,
                                      ply + 1, False)
            finally:
                self.stm ^= 1
                self.hash ^= ZSIDE
            if score >= beta:
                return beta

        moves = self._order(self._gen(False), ttmove, ply)
        bd = self.bd
        best = -MATE
        best_move = 0
        legal = 0
        pinned = None if in_check else self._pins(us)

        for mv in moves:
            frm = mv & 63
            # Only king moves, moves by pinned pieces, and moves made while in
            # check can possibly be illegal; everything else needs no check.
            risky = (in_check or bd[frm] & 7 == 5
                     or (pinned is not None and frm in pinned))
            self._make(mv)
            if risky and self._attacked(self.ksq[us], us == 0):
                self._undo(mv)
                continue
            legal += 1
            quiet = not (mv >> 12) & 15

            if legal == 1:
                score = -self._search(depth - 1, -beta, -alpha, ply + 1, True)
            else:
                reduce = 0
                if quiet and depth >= 3 and legal > 3 and not in_check:
                    reduce = 1 if legal < 7 else 2
                    if reduce >= depth:
                        reduce = depth - 1
                score = -self._search(depth - 1 - reduce, -alpha - 1, -alpha,
                                      ply + 1, True)
                if score > alpha and (reduce or beta - alpha > 1):
                    score = -self._search(depth - 1, -beta, -alpha,
                                          ply + 1, True)
            self._undo(mv)

            if score > best:
                best = score
                best_move = mv
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if quiet:
                            slot = ply << 1
                            if self.killers[slot] != mv:
                                self.killers[slot + 1] = self.killers[slot]
                                self.killers[slot] = mv
                            idx = (mv & 63) * NSQ + ((mv >> 6) & 63)
                            self.hist[idx] += depth * depth
                        break

        if legal == 0:
            if in_check:
                return -MATE + ply
            return -CONTEMPT if us == self.me else CONTEMPT

        store = best
        if store > MATE_BOUND:
            store += ply
        elif store < -MATE_BOUND:
            store -= ply
        flag = 0 if alpha_orig < best < beta else (1 if best >= beta else 2)
        if len(self.tt) > 400000:
            self.tt.clear()
        self.tt[key] = (depth, flag, store, best_move)
        return best

    # ------------------------------------------------------------------
    # root
    # ------------------------------------------------------------------
    def _emergency_move(self):
        """Last-resort move if the search ever raises.

        Uses only the engine, so it works even when our internal state is
        corrupt. Prefers the most valuable safe capture over an arbitrary
        move -- `legal[0]` could be anything, and one blunder here can turn a
        winnable game into a 0.
        """
        legal = self.engine.get_legal_moves()
        if not legal:
            return None
        if self._fallback_move is not None:
            for mv in legal:                  # last fully-searched root move
                if mv == self._fallback_move:
                    return mv
        best, best_score = legal[0], -1 << 30
        for mv in legal:
            score = 0
            if mv.piece_captured != EMPTY_SQUARE:
                score += 10 * VAL[CODE.get(mv.piece_captured, 0)]
            self.engine.make_move(mv)
            try:
                if self.engine.is_in_check():
                    score += 30           # checking the opponent is worth +2
                hangs = self.engine.get_legal_moves()
                for reply in hangs:       # avoid dropping material for free
                    if reply.end_row == mv.end_row and \
                            reply.end_col == mv.end_col:
                        score -= 9 * VAL[CODE.get(mv.piece_moved, 0)]
                        break
            finally:
                self.engine.undo_move()
            if score > best_score:
                best_score, best = score, mv
        return best

    def _budget(self):
        pool = self.clock - self.reserve
        if pool <= 0.05:
            return 0.01
        # Most games run to the 150-turn limit, but many end far earlier, so
        # plan for ~42 more moves rather than the full 75. The divisor floor
        # plus the pool-fraction cap below make flagging impossible.
        left = self.MAX_MOVES - self.moves_made
        horizon = (self.MAX_MOVES * self.MOVE_HORIZON) // 75
        if left > horizon:
            left = horizon
        if left < 8:
            left = 8
        budget = pool / left
        cap = pool * 0.15
        if budget > cap:
            budget = cap
        hard = self.TOTAL_TIME / 20.0
        if budget > hard:
            budget = hard
        if budget < 0.01:
            budget = 0.01
        return budget

    def get_best_move(self):
        started = time.perf_counter()
        self.nodes = 0
        self.nodes_expanded = 0
        try:
            move = self._pick()
        except Exception as exc:
            self.last_error = exc
            move = None
        if move is None:                       # never forfeit on a bug
            move = self._emergency_move()
        self.nodes_expanded = self.nodes
        self.clock -= time.perf_counter() - started
        self.moves_made += 1
        return move

    def _root_iter(self, depth, order, alpha, beta, us, repeats):
        """One root pass with PVS. Returns (best_move, best_score, scores)."""
        local_best = 0
        local_score = -MATE - 1
        scores = {}
        first = True
        for mv in order:
            self._make(mv)
            if self._attacked(self.ksq[us], us == 0):
                self._undo(mv)
                continue
            if first:
                score = -self._search(depth - 1, -beta, -alpha, 1, True)
            else:
                score = -self._search(depth - 1, -alpha - 1, -alpha, 1, True)
                if score > alpha:
                    score = -self._search(depth - 1, -beta, -alpha, 1, True)
            self._undo(mv)
            first = False
            if repeats[mv]:
                score -= 14              # discourage shuffling in circles
            scores[mv] = score
            if score > local_score:
                local_score = score
                local_best = mv
                if score > alpha:
                    alpha = score
        return local_best, local_score, scores

    def _pick(self):
        self._sync()
        self.me = self.stm
        self.seen.add(self.hash)

        us = self.stm
        roots = []
        for mv in self._gen(False):
            self._make(mv)
            if not self._attacked(self.ksq[us], us == 0):
                roots.append((mv, self.hash in self.seen))
            self._undo(mv)
        if not roots:
            return None
        if len(roots) == 1:
            return self._to_move(roots[0][0])

        book = _BOOK.get(self.hash)
        if book is not None:
            for mv, _ in roots:          # only if still legal here
                if mv == book:
                    return self._to_move(mv)

        # adapt greed to the situation: hoard when fragile, trade when winning
        mine, theirs = self.mat[us], self.mat[us ^ 1]
        if mine < 150 or mine + 100 < theirs:
            lw = self.LW_SAFE
        elif mine > theirs + 150:
            lw = self.LW_WIN
        else:
            lw = self.LW_MID
        if lw != self.lw:
            self.tt.clear()          # cached scores assume one greed setting
            self.lw = lw

        if self.FIXED_DEPTH:
            # deterministic mode for parameter tuning: no clock, fixed depth
            self.deadline = time.perf_counter() + 1e9
            soft = self.deadline
            max_depth = self.FIXED_DEPTH
        else:
            budget = self._budget()
            self.deadline = time.perf_counter() + budget
            soft = time.perf_counter() + budget * 0.5
            max_depth = 40

        for i in range(len(self.hist)):
            self.hist[i] >>= 2
        self.killers = [0] * 128

        order = [mv for mv, _ in roots]
        repeats = {mv: rep for mv, rep in roots}
        best_move = order[0]
        depth = 1

        prev_score = None

        while depth <= max_depth:
            try:
                if (self.USE_ASPIRATION and depth >= 4 and prev_score is not None
                        and -MATE_BOUND < prev_score < MATE_BOUND):
                    # Aspiration window: search a narrow band around the last
                    # iteration's score, widening only when it falls outside.
                    window = 45
                    while True:
                        lo = prev_score - window
                        hi = prev_score + window
                        local_best, local_score, scores = self._root_iter(
                            depth, order, lo, hi, us, repeats)
                        if lo < local_score < hi:
                            break
                        window *= 4
                        if window > 3000:
                            local_best, local_score, scores = self._root_iter(
                                depth, order, -MATE, MATE, us, repeats)
                            break
                else:
                    local_best, local_score, scores = self._root_iter(
                        depth, order, -MATE, MATE, us, repeats)
            except _TimeUp:
                break

            best_move = local_best
            prev_score = local_score
            self._fallback_move = self._to_move(best_move)
            self.depth = depth
            order.sort(key=lambda m: scores.get(m, -MATE), reverse=True)
            if local_score > MATE_BOUND:      # forced mate found, stop early
                break
            if time.perf_counter() > soft:
                break
            depth += 1

        self._sync()
        self._make(best_move)
        self.seen.add(self.hash)
        self._undo(best_move)

        chosen = self._to_move(best_move)

        # final safety net: the engine is the authority on legality
        legal = self.engine.get_legal_moves()
        for mv in legal:
            if mv == chosen:
                return chosen
        return legal[0] if legal else None
