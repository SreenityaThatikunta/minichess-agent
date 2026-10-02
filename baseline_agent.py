import time
from board import Move
from config import (
    BISHOP_PST,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EMPTY_SQUARE,
    KING_PST_LATE_GAME,
    KNIGHT_PST,
    PAWN_PST,
    PIECE_VALUES,
    ROOK_PST,
)

class BaselineAgent:
    """
    The earlier version of the agent, kept as a measurement baseline.
    Iterative-deepening alpha-beta over the engine's own move generator.
    """
    MATE_SCORE = 100000

    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 1
        self.time_limit = 0.4
        self._deadline = 0.0

    def get_best_move(self):
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        self.nodes_expanded = 0
        self._deadline = time.perf_counter() + self.time_limit
        best_move = legal_moves[0]
        best_score = self._move_order_score(best_move)

        search_depth = 1
        while time.perf_counter() < self._deadline:
            score, move, completed = self._search_root(search_depth, legal_moves)
            if not completed:
                break
            if move is not None:
                best_move = move
                best_score = score
            self.depth = search_depth
            search_depth += 1

        return best_move

    def evaluate_board(self, game_state=None):
        if game_state == "checkmate":
            return -self.MATE_SCORE if self.engine.white_to_move else self.MATE_SCORE
        if game_state == "stalemate":
            return 0

        score = 0
        tables = {
            'P': PAWN_PST,
            'N': KNIGHT_PST,
            'B': BISHOP_PST,
            'R': ROOK_PST,
            'K': KING_PST_LATE_GAME,
        }
        for row_index, row in enumerate(self.engine.board):
            for column_index, piece in enumerate(row):
                if piece == EMPTY_SQUARE:
                    continue
                score += PIECE_VALUES[piece]
                table = tables[piece[1]]
                table_row = row_index if piece[0] == 'w' else BOARD_HEIGHT - 1 - row_index
                positional_value = table[table_row][column_index]
                score += positional_value if piece[0] == 'w' else -positional_value

        if self.engine.is_in_check():
            score += -15 if self.engine.white_to_move else 15
        if self.engine.get_repetition_count() > 1:
            score += -5 if self.engine.white_to_move else 5
        return score

    def _search_root(self, depth, legal_moves):
        maximizing = self.engine.white_to_move
        alpha = -self.MATE_SCORE
        beta = self.MATE_SCORE
        best_score = -self.MATE_SCORE if maximizing else self.MATE_SCORE
        best_move = None

        for move in self._order_moves(legal_moves):
            if self._out_of_time():
                return best_score, best_move, False
            self.engine.make_move(move)
            score, completed = self._search(depth - 1, alpha, beta)
            self.engine.undo_move()
            if not completed:
                return best_score, best_move, False
            if (maximizing and score > best_score) or (not maximizing and score < best_score):
                best_score = score
                best_move = move
            if maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_move, True

    def _search(self, depth, alpha, beta):
        self.nodes_expanded += 1
        if self._out_of_time():
            return 0, False

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            if self.engine.is_in_check():
                score = -self.MATE_SCORE if self.engine.white_to_move else self.MATE_SCORE
                return score, True
            return 0, True
        if depth == 0:
            return self.evaluate_board(), True

        maximizing = self.engine.white_to_move
        best_score = -self.MATE_SCORE if maximizing else self.MATE_SCORE
        for move in self._order_moves(legal_moves):
            if self._out_of_time():
                return 0, False
            self.engine.make_move(move)
            score, completed = self._search(depth - 1, alpha, beta)
            self.engine.undo_move()
            if not completed:
                return 0, False
            if maximizing:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, True

    def _order_moves(self, moves):
        return sorted(moves, key=self._move_order_score, reverse=True)

    def _move_order_score(self, move):
        score = 0
        if move.piece_captured != EMPTY_SQUARE:
            score += abs(PIECE_VALUES[move.piece_captured]) * 10
        if move.piece_moved[1] == 'K':
            score -= 2

        if not self._out_of_time():
            self.engine.make_move(move)
            gives_check = self.engine.is_in_check()
            self.engine.undo_move()
            if gives_check:
                score += 25
        return score

    def _out_of_time(self):
        return time.perf_counter() >= self._deadline

############################################################################



## Feel free to add any additional helper methods or properties you need.



#############################################################################