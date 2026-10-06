from __future__ import annotations

import heapq
import time

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import inf
from random import Random

from .models import Direction, GameSnapshot, Position, SnakeView

class MoveStrategy(ABC):
    """Base class for every automated snake movement method."""

    @abstractmethod
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        """Return the direction for ``snake_id`` for the current turn."""


STRATEGY_REGISTRY: dict[str, type[MoveStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[MoveStrategy]], type[MoveStrategy]]:
    """Register a strategy class so it appears in the setup-screen dropdown."""

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
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        if legal:
            return rng.choice(legal)
        return snapshot.snake(snake_id).direction


@register_strategy("Greedy")
class GreedyStrategy(MoveStrategy):
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        head_x, head_y = snake.body[0]

        def distance(direction: Direction) -> int:
            dx, dy = direction.vector
            new_x, new_y = head_x + dx, head_y + dy
            return min(
                abs(new_x - apple_x) + abs(new_y - apple_y)
                for apple_x, apple_y in snapshot.apples
            )

        best_distance = min(distance(direction) for direction in legal)
        best_moves = [direction for direction in legal if distance(direction) == best_distance]
        return rng.choice(best_moves)


# Assignment template -------------------------------------------------------
# 1. Copy this class and give it a unique name.
# 2. Implement choose_move using only the immutable snapshot.
# 3. Add @register_strategy("My Strategy") above the class. It will then appear
#    automatically in the setup screen.
#
# @register_strategy("My Strategy")
# class MyStrategy(MoveStrategy):
#     def choose_move(self, snapshot, snake_id, rng):
#         legal_moves = snapshot.legal_moves_for(snake_id)
#         return legal_moves[0] if legal_moves else snapshot.snake(snake_id).direction


@register_strategy("Greedy Best-First")
class GreedyBestFirstStrategy(MoveStrategy):
    def choose_move(self, snapshot, snake_id, rng):
        import time
        start_time = time.perf_counter()
        expanded = 0
        generated = 0

        snake = snapshot.snake(snake_id)
        apples = snapshot.apples

        def h(head):
            return min(abs(head[0] - ax) + abs(head[1] - ay) for ax, ay in apples)

        # each frontier item: (body, first_move)
        frontier = [(snake.body, None)]
        visited = {snake.body[0]}
        result = None

        while frontier and apples and result is None:
            frontier.sort(key=lambda item: h(item[0][0]))
            body, first_move = frontier.pop(0)
            expanded += 1

            for move in Direction:
                dx, dy = move.vector
                new_head = (body[0][0] + dx, body[0][1] + dy)
                if not (0 <= new_head[0] < snapshot.columns and 0 <= new_head[1] < snapshot.rows):
                    continue
                if new_head in body[:-1]:
                    continue
                new_body = (new_head,) + body[:-1]
                if new_head in visited:
                    continue
                visited.add(new_head)
                generated += 1

                if first_move is None:
                    first_move_here = move
                else:
                    first_move_here = first_move
                if new_head in apples:
                    result = first_move_here
                    break
                frontier.append((new_body, first_move_here))

        if result is None:
            legal_moves = snapshot.legal_moves_for(snake_id)
            result = legal_moves[0] if legal_moves else snake.direction

        elapsed = time.perf_counter() - start_time
        print(f"Greedy Best-First: expanded={expanded}, generated={generated}, time={elapsed:.4f}s")
        return result


@register_strategy("A*")
class AStarStrategy(MoveStrategy):
    def choose_move(self, snapshot, snake_id, rng):
        import time
        start_time = time.perf_counter()
        expanded = 0
        generated = 0

        snake = snapshot.snake(snake_id)
        apples = snapshot.apples

        def h(head):
            return min(abs(head[0] - ax) + abs(head[1] - ay) for ax, ay in apples)

        # each frontier item: (body, first_move, g)
        frontier = [(snake.body, None, 0)]
        visited = set()
        result = None

        while frontier and apples:
            frontier.sort(key=lambda item: item[2] + h(item[0][0]))
            body, first_move, g = frontier.pop(0)
            if body[0] in visited:
                continue
            visited.add(body[0])

            if body[0] in apples:
                result = first_move
                break
            expanded += 1

            for move in Direction:
                dx, dy = move.vector
                new_head = (body[0][0] + dx, body[0][1] + dy)
                if not (0 <= new_head[0] < snapshot.columns and 0 <= new_head[1] < snapshot.rows):
                    continue
                if new_head in body[:-1]:
                    continue
                new_body = (new_head,) + body[:-1]
                if new_head in visited:
                    continue
                generated += 1

                if first_move is None:
                    first_move_here = move
                else:
                    first_move_here = first_move
                frontier.append((new_body, first_move_here, g + 1))

        if result is None:
            legal_moves = snapshot.legal_moves_for(snake_id)
            result = legal_moves[0] if legal_moves else snake.direction

        elapsed = time.perf_counter() - start_time
        print(f"A*: expanded={expanded}, generated={generated}, time={elapsed:.4f}s")
        return result


@register_strategy("UCS")
class UCSStrategy(MoveStrategy):
    def __init__(self):
        self.total_expanded = 0
        self.total_generated = 0
        self.total_time = 0.0
        self.total_searches = 0

    def choose_move(self, snapshot, snake_id, rng):
        start_time = time.perf_counter()
        expanded = generated = 0

        snake = snapshot.snake(snake_id)
        apples = set(snapshot.apples)
        legal_moves = snapshot.legal_moves_for(snake_id)

        if not legal_moves or not apples:
            return snake.direction

        # (cost, order, body, first_move)
        frontier = [(0, 0, snake.body, None)]
        visited = set()
        order = 1
        result = None
        result_cost = 0

        while frontier:
            cost, _, body, first_move = heapq.heappop(frontier)
            head = body[0]

            if head in visited:
                continue

            visited.add(head)
            expanded += 1

            if head in apples:
                result, result_cost = first_move, cost
                break

            for move in Direction:
                dx, dy = move.vector
                new_head = (head[0] + dx, head[1] + dy)
                x, y = new_head

                if not (0 <= x < snapshot.columns and 0 <= y < snapshot.rows):
                    continue

                new_body = (
                    (new_head,) + body
                    if new_head in apples
                    else (new_head,) + body[:-1]
                )

                if new_head in new_body[1:] or new_head in visited:
                    continue

                heapq.heappush(
                    frontier,
                    (cost + 1, order, new_body, first_move or move),
                )
                order += 1
                generated += 1

        if result is None:
            result = legal_moves[0]

        elapsed = time.perf_counter() - start_time

        self.total_expanded += expanded
        self.total_generated += generated
        self.total_time += elapsed
        self.total_searches += 1

        print(
            f"UCS: expanded={expanded}, generated={generated}, "
            f"path_cost={result_cost}, time={elapsed:.6f}s | "
            f"TOTAL: searches={self.total_searches}, "
            f"expanded={self.total_expanded}, "
            f"generated={self.total_generated}, "
            f"time={self.total_time:.6f}s"
        )

        return result

    @register_strategy("IDS")
    class IDSStrategy(MoveStrategy):
        def __init__(self):
            self.total_expanded = 0
            self.total_generated = 0
            self.total_time = 0.0
            self.total_searches = 0

        def choose_move(self, snapshot, snake_id, rng):
            start_time = time.perf_counter()
            expanded = generated = 0

            snake = snapshot.snake(snake_id)
            apples = set(snapshot.apples)
            legal_moves = snapshot.legal_moves_for(snake_id)

            if not legal_moves or not apples:
                return snake.direction

            def dls(body, depth, limit, first_move, visited):
                nonlocal expanded, generated

                head = body[0]
                if head in apples:
                    return first_move
                if depth == limit:
                    return None

                visited.add(head)
                expanded += 1

                for move in Direction:
                    dx, dy = move.vector
                    new_head = (head[0] + dx, head[1] + dy)
                    x, y = new_head

                    if not (0 <= x < snapshot.columns and 0 <= y < snapshot.rows):
                        continue

                    new_body = (
                        (new_head,) + body
                        if new_head in apples
                        else (new_head,) + body[:-1]
                    )

                    if new_head in new_body[1:] or new_head in visited:
                        continue

                    generated += 1
                    found = dls(
                        new_body, depth + 1, limit,
                                  first_move or move, visited
                    )

                    if found is not None:
                        return found

                return None

            result = None
            result_depth = 0
            max_depth = snapshot.rows * snapshot.columns

            for limit in range(max_depth + 1):
                result = dls(snake.body, 0, limit, None, set())
                if result is not None:
                    result_depth = limit
                    break

            if result is None:
                result = legal_moves[0]

            elapsed = time.perf_counter() - start_time

            self.total_expanded += expanded
            self.total_generated += generated
            self.total_time += elapsed
            self.total_searches += 1

            print(
                f"IDS: expanded={expanded}, generated={generated}, "
                f"path_cost={result_depth}, time={elapsed:.6f}s | "
                f"TOTAL: searches={self.total_searches}, "
                f"expanded={self.total_expanded}, "
                f"generated={self.total_generated}, "
                f"time={self.total_time:.6f}s"
            )

            return result

        @register_strategy("Minimax")
        class MinimaxStrategy(MoveStrategy):
            MAX_DEPTH = 3

            def choose_move(
                    self, snapshot: GameSnapshot, snake_id: str, rng: Random
            ) -> Direction:
                start_time = time.perf_counter()
                states_generated = 0
                states_evaluated = 0

                legal_moves = snapshot.legal_moves_for(snake_id)
                snake = snapshot.snake(snake_id)

                if not legal_moves:
                    return snake.direction

                opponent = next(
                    (
                        other
                        for other in snapshot.snakes
                        if other.snake_id != snake_id
                    ),
                    None,
                )

                if opponent is None:
                    return legal_moves[0]

                opponent_id = opponent.snake_id

                snakes = {
                    current.snake_id: {
                        "body": tuple(current.body),
                        "direction": current.direction,
                        "score": current.score,
                    }
                    for current in snapshot.snakes
                }
                apples = set(snapshot.apples)

                def evaluate(state, remaining_apples):
                    me = state[snake_id]
                    other = state[opponent_id]

                    score_difference = me["score"] - other["score"]

                    if not remaining_apples:
                        return score_difference * 1000

                    my_head = me["body"][0]
                    other_head = other["body"][0]

                    my_distance = min(
                        abs(my_head[0] - x) + abs(my_head[1] - y)
                        for x, y in remaining_apples
                    )
                    other_distance = min(
                        abs(other_head[0] - x) + abs(other_head[1] - y)
                        for x, y in remaining_apples
                    )

                    return (
                            score_difference * 100
                            + other_distance
                            - my_distance
                    )

                def get_legal_moves(state, remaining_apples, current_id):
                    current = state[current_id]
                    body = current["body"]
                    head = body[0]
                    moves = []

                    for move in Direction:
                        if (
                                len(body) > 1
                                and move is current["direction"].opposite
                        ):
                            continue

                        dx, dy = move.vector
                        new_head = (head[0] + dx, head[1] + dy)
                        x, y = new_head

                        if not (
                                0 <= x < snapshot.columns
                                and 0 <= y < snapshot.rows
                        ):
                            continue

                        growing = new_head in remaining_apples
                        own_body = body if growing else body[:-1]

                        if new_head in own_body:
                            continue

                        hits_other_snake = any(
                            new_head in other["body"]
                            for other_id, other in state.items()
                            if other_id != current_id
                        )

                        if hits_other_snake:
                            continue

                        moves.append(move)

                    return moves

                def apply_move(state, remaining_apples, current_id, move):
                    new_state = {
                        sid: {
                            "body": tuple(data["body"]),
                            "direction": data["direction"],
                            "score": data["score"],
                        }
                        for sid, data in state.items()
                    }
                    new_apples = set(remaining_apples)

                    current = new_state[current_id]
                    body = current["body"]

                    dx, dy = move.vector
                    head_x, head_y = body[0]
                    new_head = (head_x + dx, head_y + dy)

                    if new_head in new_apples:
                        new_body = (new_head,) + body
                        new_apples.remove(new_head)
                        new_score = current["score"] + 1
                    else:
                        new_body = (new_head,) + body[:-1]
                        new_score = current["score"]

                    new_state[current_id] = {
                        "body": new_body,
                        "direction": move,
                        "score": new_score,
                    }

                    return new_state, new_apples

                def minimax(state, remaining_apples, current_id, depth):
                    nonlocal states_generated, states_evaluated

                    if depth == 0 or not remaining_apples:
                        states_evaluated += 1
                        return evaluate(state, remaining_apples)

                    moves = get_legal_moves(
                        state,
                        remaining_apples,
                        current_id,
                    )

                    if not moves:
                        states_evaluated += 1

                        if current_id == snake_id:
                            return -10000

                        return 10000

                    next_id = (
                        opponent_id
                        if current_id == snake_id
                        else snake_id
                    )

                    if current_id == snake_id:
                        best_value = float("-inf")

                        for move in moves:
                            next_state, next_apples = apply_move(
                                state,
                                remaining_apples,
                                current_id,
                                move,
                            )
                            states_generated += 1

                            value = minimax(
                                next_state,
                                next_apples,
                                next_id,
                                depth - 1,
                            )

                            best_value = max(best_value, value)

                        return best_value

                    best_value = float("inf")

                    for move in moves:
                        next_state, next_apples = apply_move(
                            state,
                            remaining_apples,
                            current_id,
                            move,
                        )
                        states_generated += 1

                        value = minimax(
                            next_state,
                            next_apples,
                            next_id,
                            depth - 1,
                        )

                        best_value = min(best_value, value)

                    return best_value

                best_move = legal_moves[0]
                best_value = float("-inf")

                for move in legal_moves:
                    next_state, next_apples = apply_move(
                        snakes,
                        apples,
                        snake_id,
                        move,
                    )
                    states_generated += 1

                    value = minimax(
                        next_state,
                        next_apples,
                        opponent_id,
                        self.MAX_DEPTH - 1,
                    )

                    if value > best_value:
                        best_value = value
                        best_move = move

                elapsed = time.perf_counter() - start_time

                print(
                    f"Minimax: depth={self.MAX_DEPTH}, "
                    f"generated={states_generated}, "
                    f"evaluated={states_evaluated}, "
                    f"time={elapsed:.6f}s"
                )

                return best_move


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
