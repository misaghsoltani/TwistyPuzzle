#!/usr/bin/env python3
"""Measure how well the extension scales across Python threads.

Long-running operations release the Python GIL during execution, allowing
puzzle construction and rendering to execute concurrently across threads on
standard GIL builds as well as free-threaded builds. This script benchmarks
threading throughput and compares thread-pool execution against
``build_many`` and ``render_many``, which parallelize internally in native Rust.

    python scripts/bench_threads.py
    python scripts/bench_threads.py --puzzles 24 --threads 1 2 4 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from typing import TYPE_CHECKING

import twistypuzzle as tp

if TYPE_CHECKING:
    from collections.abc import Callable


def timed(fn: Callable[[], None], repeat: int) -> float:
    """Best of `repeat` runs, in seconds."""
    return min(_one(fn) for _ in range(repeat))


def _one(fn: Callable[[], None]) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print results."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", type=int, default=24, help="how many puzzles per round")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="thread counts to try")
    ap.add_argument("--repeat", type=int, default=2, help="rounds per measurement, best taken")
    ap.add_argument("--size", type=int, default=256, help="render size in pixels")
    args = ap.parse_args(argv)

    # The heaviest puzzles, so the parallel region dominates the measurement
    # instead of the thread-pool overhead around it.
    entries = tp.catalog_entries()
    survey = tp.build_many([e.recipe for e in entries])
    ranked = sorted(zip(entries, survey, strict=True), key=lambda x: -x[1].piece_count)
    recipes = [e.recipe for e, _ in ranked[: args.puzzles]]
    pieces = sum(p.piece_count for _, p in ranked[: args.puzzles])

    gil = "off (free-threaded build)" if tp.free_threaded else "on"
    print(f"python {sys.version.split()[0]}   GIL: {gil}   rayon threads: {tp.thread_count()}")
    print(f"{len(recipes)} of the heaviest puzzles ({pieces} pieces), rendered at {args.size}x{args.size}\n")

    def build_pool(n: int) -> None:
        """Build puzzles in parallel on `n` threads."""
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(tp.Puzzle, recipes))

    def render_pool(n: int, puzzles: list[tp.Puzzle]) -> None:
        """Render puzzles in parallel on `n` threads."""
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda p: p.render(args.size, args.size), puzzles))

    def analyze(p: tp.Puzzle) -> int:
        """Render, then walk the pixels in Python.

        The render operation releases the GIL, enabling multi-threaded execution
        on both standard and free-threaded CPython builds. In contrast, the subsequent
        pixel processing loop executes purely in Python, contending on the GIL in
        standard CPython while scaling concurrently under free-threading.
        """
        pixels = p.render(args.size, args.size).to_bytes()
        # Deliberately a plain Python loop over every pixel: that is the shape
        # of a caller's own post-processing, and it is what a GIL serializes.
        total = 0
        for i in range(0, len(pixels), 4):
            total += pixels[i] + pixels[i + 1] + pixels[i + 2]
        return total

    def pipeline_pool(n: int, puzzles: list[tp.Puzzle]) -> None:
        """Render and then post-process in Python, on `n` threads."""
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(analyze, puzzles))

    puzzles: list[tp.Puzzle] = tp.build_many(recipes)

    print(
        f"{'threads':>8}  {'build (s)':>10} {'up':>7}   {'render (s)':>10} {'up':>7}   {'render+py (s)':>14} {'up':>7}"
    )
    base_b = base_r = base_p = None
    for n in args.threads:
        b = timed(lambda n=n: build_pool(n), args.repeat)
        r = timed(lambda n=n: render_pool(n, puzzles), args.repeat)
        p = timed(lambda n=n: pipeline_pool(n, puzzles), args.repeat)
        base_b = base_b or b
        base_r = base_r or r
        base_p = base_p or p
        print(
            f"{n:>8}  {b:>10.3f} {base_b / b:>6.2f}x   {r:>10.3f} {base_r / r:>6.2f}x   {p:>14.3f} {base_p / p:>6.2f}x"
        )

    b = timed(lambda: tp.build_many(recipes), args.repeat)
    r = timed(lambda: tp.render_many(recipes, args.size, args.size, supersample=2), args.repeat)
    print(f"\n{'build_many':>12}: {b:.3f}s  ({base_b / b:.2f}x a single thread)")
    print(f"{'render_many':>12}: {r:.3f}s  (builds and renders in one pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
