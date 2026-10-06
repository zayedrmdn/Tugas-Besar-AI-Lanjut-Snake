from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import inf
from random import Random

from .models import Direction, GameSnapshot, Position, SnakeView


class MoveStrategy(ABC):
    @abstractmethod
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        """Return the direction for snake_id for the current turn."""


STRATEGY_REGISTRY: dict[str, type[MoveStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[MoveStrategy]], type[MoveStrategy]]:
    def decorator(strategy_class: type[MoveStrategy]) -> type[MoveStrategy]:
        if not name.strip():
            raise ValueError("Strategy names cannot be empty.")
        STRATEGY_REGISTRY[name] = strategy_class
        return strategy_class

    return decorator


def create_strategy(name: str) -> MoveStrategy:
    try:
        return STRATEGY_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc


@register_strategy("Safe Random")
class SafeRandomStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        return rng.choice(legal) if legal else snapshot.snake(snake_id).direction


@register_strategy("Greedy")
class GreedyStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        hx, hy = snake.body[0]
        distances = {}
        for direction in legal:
            dx, dy = direction.vector
            nx, ny = hx + dx, hy + dy
            distances[direction] = min(
                abs(nx - ax) + abs(ny - ay) for ax, ay in snapshot.apples
            )
        best = min(distances.values())
        return rng.choice([d for d in legal if distances[d] == best])


# ---------------------------------------------------------------------------
# BFS - Single Player
# ---------------------------------------------------------------------------


def _inside(snapshot: GameSnapshot, p: Position) -> bool:
    x, y = p
    return 0 <= x < snapshot.columns and 0 <= y < snapshot.rows


def _next_position(p: Position, direction: Direction) -> Position:
    dx, dy = direction.vector
    return p[0] + dx, p[1] + dy


def _direction_between(a: Position, b: Position) -> Direction:
    dx, dy = b[0] - a[0], b[1] - a[1]
    for direction in Direction:
        if direction.vector == (dx, dy):
            return direction
    raise ValueError("Positions are not adjacent")


@register_strategy("BFS")
class BFSStrategy(MoveStrategy):
    """Breadth-First Search toward the nearest reachable apple.

    The search treats all snake bodies as obstacles. If no apple is reachable,
    it chooses the legal move leading to the largest reachable free area.
    """

    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal:
            return snake.direction
        if not snapshot.apples:
            return legal[0]

        start = snake.body[0]
        obstacles = {part for s in snapshot.snakes for part in s.body}
        # The current head is the BFS start, so it must not be treated as an obstacle.
        obstacles.discard(start)

        queue = deque([start])
        parent: dict[Position, Position | None] = {start: None}
        target: Position | None = None

        # Deterministic order makes demonstrations/tests reproducible.
        directions = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)

        while queue:
            current = queue.popleft()
            if current in snapshot.apples:
                target = current
                break

            for direction in directions:
                nxt = _next_position(current, direction)
                if not _inside(snapshot, nxt) or nxt in obstacles or nxt in parent:
                    continue
                parent[nxt] = current
                queue.append(nxt)

        if target is not None:
            # Walk backward from apple to head and return the first step.
            current = target
            while parent[current] != start:
                current = parent[current]  # type: ignore[assignment]
            return _direction_between(start, current)

        # No apple is reachable. Prefer a move with the largest connected area.
        best_move = legal[0]
        best_area = -1
        for move in legal:
            next_head = _next_position(start, move)
            area = self._reachable_area(snapshot, next_head, obstacles)
            if area > best_area:
                best_area = area
                best_move = move
        return best_move

    def _reachable_area(
        self, snapshot: GameSnapshot, start: Position, obstacles: set[Position]
    ) -> int:
        if not _inside(snapshot, start) or start in obstacles:
            return 0
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            for direction in Direction:
                nxt = _next_position(current, direction)
                if _inside(snapshot, nxt) and nxt not in obstacles and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return len(visited)


# ---------------------------------------------------------------------------
# Alpha-Beta Pruning - Multiplayer / AI vs AI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSnake:
    body: tuple[Position, ...]
    direction: Direction
    score: int


@dataclass(frozen=True)
class SearchState:
    snakes: tuple[tuple[str, SearchSnake], ...]
    apples: frozenset[Position]
    active_id: str

    def get_snake(self, snake_id: str) -> SearchSnake:
        for sid, snake in self.snakes:
            if sid == snake_id:
                return snake
        raise KeyError(snake_id)


@register_strategy("Alpha-Beta")
class AlphaBetaStrategy(MoveStrategy):
    """Depth-limited minimax with Alpha-Beta pruning.

    In AI-vs-AI mode the other snake is modeled as the minimizing player.
    The evaluation rewards score, proximity to apples, mobility, and open space.
    """

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.root_id = ""
        self.opponent_id = ""
        self._rows = 0
        self._columns = 0

    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal:
            return snake.direction

        # Alpha-Beta is most meaningful when there are two snakes.
        if len(snapshot.snakes) < 2:
            return self._best_single_move(snapshot, snake_id, legal)

        self.root_id = snake_id
        self._rows = snapshot.rows
        self._columns = snapshot.columns
        self.opponent_id = next(s.snake_id for s in snapshot.snakes if s.snake_id != snake_id)
        state = self._make_state(snapshot)

        best_score = -inf
        best_moves: list[Direction] = []
        alpha, beta = -inf, inf

        for move in legal:
            child = self._apply_move(state, snake_id, move, snapshot)
            score = self._alphabeta(child, self.depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, best_score)

        return rng.choice(best_moves or list(legal))

    def _best_single_move(self, snapshot: GameSnapshot, snake_id: str, legal: tuple[Direction, ...]) -> Direction:
        snake = snapshot.snake(snake_id)
        if not snapshot.apples:
            return legal[0]
        hx, hy = snake.body[0]
        return min(
            legal,
            key=lambda d: min(
                abs(hx + d.vector[0] - ax) + abs(hy + d.vector[1] - ay)
                for ax, ay in snapshot.apples
            ),
        )

    def _make_state(self, snapshot: GameSnapshot) -> SearchState:
        snakes = tuple(
            (s.snake_id, SearchSnake(s.body, s.direction, s.score))
            for s in snapshot.snakes
        )
        return SearchState(snakes, frozenset(snapshot.apples), snapshot.active_snake_id)

    def _legal_moves(self, state: SearchState, snake_id: str, snapshot: GameSnapshot) -> tuple[Direction, ...]:
        snake = state.get_snake(snake_id)
        legal: list[Direction] = []
        for direction in Direction:
            if len(snake.body) > 1 and direction is snake.direction.opposite:
                continue
            new_head = _next_position(snake.body[0], direction)
            if self._collision_with_snapshot(state, snake_id, new_head, new_head in state.apples, snapshot) is None:
                legal.append(direction)
        return tuple(legal)

    def _apply_move(self, state: SearchState, snake_id: str, direction: Direction, snapshot: GameSnapshot) -> SearchState:
        # Use the real board dimensions from snapshot. This method intentionally
        # mirrors GameEngine.step's sequential movement semantics.
        snake = state.get_snake(snake_id)
        new_head = _next_position(snake.body[0], direction)
        growing = new_head in state.apples

        if self._collision_with_snapshot(state, snake_id, new_head, growing, snapshot) is not None:
            # Illegal branches are normally never generated. Keep this defensive.
            return state

        new_body = (new_head,) + (snake.body if growing else snake.body[:-1])
        new_score = snake.score + (1 if growing else 0)
        new_snakes = tuple(
            (sid, SearchSnake(new_body, direction if sid == snake_id else s.direction, new_score if sid == snake_id else s.score))
            for sid, s in state.snakes
        )
        apples = state.apples - {new_head} if growing else state.apples
        next_id = self.opponent_id if snake_id == self.root_id else self.root_id
        return SearchState(new_snakes, frozenset(apples), next_id)

    def _collision_with_snapshot(
        self, state: SearchState, snake_id: str, new_head: Position, growing: bool, snapshot: GameSnapshot
    ) -> str | None:
        x, y = new_head
        if x < 0 or x >= snapshot.columns or y < 0 or y >= snapshot.rows:
            return "wall"
        moving = state.get_snake(snake_id)
        own = moving.body if growing else moving.body[:-1]
        if new_head in own:
            return "self"
        for other_id, other in state.snakes:
            if other_id != snake_id and new_head in other.body:
                return "other"
        return None

    def _alphabeta(self, state: SearchState, depth: int, alpha: float, beta: float) -> float:
        if depth <= 0 or not state.apples:
            return self._evaluate(state)

        legal = self._legal_moves(state, state.active_id, self._current_snapshot_stub(state))
        if not legal:
            return -10_000 if state.active_id == self.root_id else 10_000

        if state.active_id == self.root_id:
            value = -inf
            for move in legal:
                child = self._apply_move(state, state.active_id, move, self._current_snapshot_stub(state))
                value = max(value, self._alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = inf
        for move in legal:
            child = self._apply_move(state, state.active_id, move, self._current_snapshot_stub(state))
            value = min(value, self._alphabeta(child, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _current_snapshot_stub(self, state: SearchState) -> GameSnapshot:
        # Only dimensions are needed by search; the dimensions are injected by
        # choose_move into the instance before recursion.
        return GameSnapshot(
            rows=self._rows,
            columns=self._columns,
            snakes=tuple(
                SnakeView(sid, s.body, s.direction, s.score)
                for sid, s in state.snakes
            ),
            apples=tuple(state.apples),
            rounds_completed=0,
            rounds_remaining=999,
            active_snake_id=state.active_id,
            legal_moves=(),
        )

    def _evaluate(self, state: SearchState) -> float:
        me = state.get_snake(self.root_id)
        opponent = state.get_snake(self.opponent_id)
        score = 100.0 * (me.score - opponent.score)

        if state.apples:
            my_dist = min(abs(me.body[0][0] - x) + abs(me.body[0][1] - y) for x, y in state.apples)
            op_dist = min(abs(opponent.body[0][0] - x) + abs(opponent.body[0][1] - y) for x, y in state.apples)
            score += 3.0 * (op_dist - my_dist)

        score += 1.5 * (len(self._legal_moves(state, self.root_id, self._current_snapshot_stub(state))) - len(self._legal_moves(state, self.opponent_id, self._current_snapshot_stub(state))))
        return score
