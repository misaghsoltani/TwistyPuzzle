#!/usr/bin/env python3
"""Draw every cataloged puzzle onto one labeled contact sheet.

Builds all eighty-five puzzles across every core, renders them from Python
threads (each render hands the interpreter back, so the threads really do run
at once), groups them by shell family and lays them out in a captioned grid.

Nothing outside the standard library is needed: the thumbnails, the
compositing and the text all come from ``twistypuzzle`` itself.

    python scripts/catalog_sheet.py -o catalog.png

``--scramble N`` turns each puzzle before photographing it, ``--tile`` changes
the thumbnail size and ``--columns`` reshapes the page.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import twistypuzzle as tp

# --- palette ---------------------------------------------------------------
PAGE = (247, 247, 249, 255)
CARD = (255, 255, 255, 255)
RULE = (223, 226, 233, 255)
INK = (24, 26, 34, 255)
MUTED = (110, 116, 130, 255)
BAND = (235, 238, 244, 255)
ACCENT = (37, 82, 178, 255)

# --- metrics, in pixels ----------------------------------------------------
MARGIN = 40
GUTTER = 16
CARD_PAD = 8
CAPTION = 46  # label strip under each thumbnail
BAND_H = 52
TITLE_H = 132
FOOT_H = 64
SS = 2  # text is drawn at this multiple and averaged back down

# A three-quarter view: head-on, the flatter puzzles are indistinguishable
# silhouettes. The distance keeps a unit-radius puzzle inside the frame with
# room to spare at this field of view.
YAW, PITCH, DISTANCE = -28.0, 20.0, 8.4


def text_tile(text: str, scale: int, color: tuple[int, int, int, int], bg: tuple[int, int, int, int]) -> tp.Image:
    """One line of text, antialiased by drawing large and averaging down."""
    w, h = tp.Image.text_size(text, scale * SS)
    # `downsample` truncates, so hand it whole blocks to average.
    img = tp.Image(w + (-w % SS) + SS, h + (-h % SS), bg)
    img.draw_text(0, 0, text, color, scale * SS)
    return img.downsample(SS)


def put_text(
    sheet: tp.Image,
    x: int,
    y: int,
    text: str,
    scale: int,
    color: tuple[int, int, int, int],
    bg: tuple[int, int, int, int],
    center_in: int = 0,
) -> None:
    """Draw a line of text onto the sheet, optionally centered in a box."""
    if not text:
        return
    tile = text_tile(text, scale, color, bg)
    if center_in:
        x += (center_in - tile.width) // 2
    sheet.blit(tile, x, y)


def label_for(entry: tp.CatalogEntry) -> str:
    """A caption that identifies the puzzle.

    Ten catalog entries carry the placeholder name ``"Unknown"``. Those get
    their recipe instead, trimmed to the shell and cut terms, which is short
    enough to read and unique where the name is not.
    """
    if entry.name.strip().lower() != "unknown":
        return entry.name
    terms = [t.split("=", 1)[-1] for t in entry.recipe.removeprefix("?").split("&")]
    return " / ".join(terms)


def build_sheet(args: argparse.Namespace) -> tuple[tp.Image, float, float]:
    """Build a contact sheet of every puzzle in the catalog."""
    tile, cols = args.tile, max(1, args.columns)
    cell_w = tile + 2 * CARD_PAD
    cell_h = tile + 2 * CARD_PAD + CAPTION

    entries = tp.catalog_entries()
    families: dict[str, list[tp.CatalogEntry]] = {}
    for e in entries:
        families.setdefault(e.family, []).append(e)

    # --- build and photograph, a chunk at a time --------------------------
    #
    # A chunk rather than the whole catalog at once, because a built puzzle
    # is much larger than a picture of it: holding all eighty-five costs about
    # 190 MB, while the thumbnails together are under twenty. Each chunk is
    # several times the thread count, so the cores stay busy while only a
    # handful of puzzles are ever alive.
    chunk_size = max(4, tp.thread_count())
    build_s = render_s = 0.0
    counts: dict[str, int] = {}
    images: dict[str, tp.Image] = {}

    def shoot(item: tuple[int, tp.Puzzle]) -> tp.Image:
        index, p = item
        p.background = CARD
        p.supersample = args.supersample
        p.show_edges = True
        p.show_arrows = False
        p.look_from(YAW, PITCH, DISTANCE)
        if args.scramble:
            # Seeded per puzzle, so the sheet is the same however the work is
            # scheduled.
            p.seed(args.seed + index)
            p.scramble(args.scramble)
            p.settle()
        return p.render(tile, tile)

    with ThreadPoolExecutor(max_workers=tp.thread_count()) as pool:
        for start in range(0, len(entries), chunk_size):
            chunk = entries[start : start + chunk_size]
            t0 = time.perf_counter()
            puzzles = tp.build_many([e.recipe for e in chunk])
            build_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            shots = list(pool.map(shoot, enumerate(puzzles, start)))
            render_s += time.perf_counter() - t0

            for e, p, shot in zip(chunk, puzzles, shots, strict=True):
                counts[e.recipe] = p.piece_count
                images[e.recipe] = shot
            # The puzzles go out of scope here, and only the pictures are kept.
            del puzzles, shots

    # --- page size -------------------------------------------------------
    width = MARGIN * 2 + cols * cell_w + (cols - 1) * GUTTER
    height = MARGIN + TITLE_H
    for members in families.values():
        rows = -(-len(members) // cols)
        height += BAND_H + GUTTER + rows * cell_h + (rows - 1) * GUTTER + GUTTER * 2
    height += FOOT_H + MARGIN

    # --- compose ---------------------------------------------------------
    sheet = tp.Image(width, height, PAGE)
    put_text(sheet, MARGIN, MARGIN + 4, "The twistypuzzle catalog", 5, INK, PAGE)
    put_text(
        sheet,
        MARGIN,
        MARGIN + 58,
        f"{len(entries)} puzzles in {len(families)} families, {sum(counts.values())} pieces, every coordinate exact",
        2,
        MUTED,
        PAGE,
    )
    sheet.fill_rect(MARGIN, MARGIN + TITLE_H - 26, width - 2 * MARGIN, 2, RULE)

    y = MARGIN + TITLE_H
    for family, members in families.items():
        sheet.fill_rect(MARGIN, y, width - 2 * MARGIN, BAND_H, BAND)
        put_text(sheet, MARGIN + 14, y + 15, family.upper(), 3, ACCENT, BAND)
        tally = f"{len(members)} puzzle{'s' if len(members) != 1 else ''}"
        tw, _ = tp.Image.text_size(tally, 2)
        put_text(sheet, width - MARGIN - 14 - tw, y + 19, tally, 2, MUTED, BAND)
        y += BAND_H + GUTTER

        for i, entry in enumerate(members):
            x = MARGIN + (i % cols) * (cell_w + GUTTER)
            cy = y + (i // cols) * (cell_h + GUTTER)

            sheet.fill_rect(x, cy, cell_w, cell_h, CARD)
            for rx, ry, rw, rh in (
                (x, cy, cell_w, 1),
                (x, cy + cell_h - 1, cell_w, 1),
                (x, cy, 1, cell_h),
                (x + cell_w - 1, cy, 1, cell_h),
            ):
                sheet.fill_rect(rx, ry, rw, rh, RULE)
            sheet.blit(images[entry.recipe], x + CARD_PAD, cy + CARD_PAD)

            ty = cy + CARD_PAD + tile + 5
            put_text(
                sheet, x + CARD_PAD, ty, tp.Image.fit_text(label_for(entry), tile, 2), 2, INK, CARD, center_in=tile
            )
            meta = f"{entry.kind}, {counts[entry.recipe]} pieces"
            put_text(sheet, x + CARD_PAD, ty + 21, tp.Image.fit_text(meta, tile, 1), 1, MUTED, CARD, center_in=tile)

        rows = -(-len(members) // cols)
        y += rows * cell_h + (rows - 1) * GUTTER + GUTTER * 2

    foot: str = f"twistypuzzle {tp.__version__} | {len(entries)} puzzles, {sum(counts.values())} pieces"
    sheet.fill_rect(MARGIN, height - MARGIN - FOOT_H + 10, width - 2 * MARGIN, 2, RULE)
    put_text(sheet, MARGIN, height - MARGIN - FOOT_H + 26, foot, 2, MUTED, PAGE)
    return sheet, build_s, render_s


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the catalog sheet script."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=Path("docs/catalog.png"))
    ap.add_argument("--tile", type=int, default=224, help="thumbnail size in pixels")
    ap.add_argument("--columns", type=int, default=8, help="puzzles per row")
    ap.add_argument("--supersample", type=int, default=3, help="thumbnail antialiasing factor")
    ap.add_argument("--scramble", type=int, default=0, help="random moves before photographing")
    ap.add_argument("--seed", type=int, default=1, help="seed for --scramble")
    args = ap.parse_args(argv)

    sheet, build_s, render_s = build_sheet(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(args.output))
    print(
        f"built in {build_s:.2f}s, rendered in {render_s:.2f}s on {tp.thread_count()} threads\n"
        f"wrote {args.output} ({sheet.width}x{sheet.height})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
