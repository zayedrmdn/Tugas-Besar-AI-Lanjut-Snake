from __future__ import annotations

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
