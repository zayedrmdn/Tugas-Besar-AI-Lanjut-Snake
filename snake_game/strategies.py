from __future__ import annotations

import heapq
import time

from abc import ABC, abstractmethod
from collections.abc import Callable
from random import Random

from .models import Direction, GameSnapshot

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