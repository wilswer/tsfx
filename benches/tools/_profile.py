"""Read a samply profile and its symbol sidecar into resolved call stacks.

Record with ``samply record --save-only --unstable-presymbolicate -o NAME.json.gz``;
samply writes ``NAME.json.gz`` and ``NAME.json.syms.json`` next to each other.
"""

import bisect
import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Leaf functions of a worker thread that is waiting rather than working
# (macOS and Linux names). macOS samples sleeping threads; Linux samples only
# running ones, so there idle time is rayon spinning while it looks for work.
IDLE_LEAVES = (
    "swtch_pri",
    "__psynch_cvwait",
    "sched_yield",
    "futex",
    "syscall",
    "crossbeam_epoch",
    "crossbeam_deque",
)
COPY_LEAVES = ("memmove", "memcpy")
ALLOC_LEAVES = ("malloc", "free", "realloc")
_TSFX_FUNCTION = re.compile(r"tsfx::(?:feature_extractors|utils)::[A-Za-z_:]+[A-Za-z_]")


def _clean(name: str) -> str:
    """Drop the symbol hash and shorten ``<T as Trait>`` to ``T``."""
    name = re.sub(r"::h[0-9a-f]{16}$", "", name)
    return re.sub(r"<([^<>]*?) as [^<>]*?>", r"\1", name)


class _Symbols:
    def __init__(self, syms: dict[str, Any]) -> None:
        strings = syms["string_table"]
        self._libs: dict[str, tuple[list[int], list[tuple[int, int, str]]]] = {}
        for lib in syms["data"]:
            rows = sorted(
                (e["rva"], e["size"], strings[e["symbol"]]) for e in lib["symbol_table"]
            )
            self._libs[lib["debug_name"]] = ([r[0] for r in rows], rows)

    def lookup(self, lib: str, address: int) -> str:
        if lib not in self._libs:
            return f"<{lib}>"
        starts, rows = self._libs[lib]
        i = bisect.bisect_right(starts, address) - 1
        if i >= 0 and address < rows[i][0] + rows[i][1]:
            return _clean(rows[i][2])
        return f"<{lib}+{address:#x}>"


def _frame_names(thread: dict, libs: list, symbols: _Symbols) -> list[str]:
    """Resolve every frame of a thread to a function name."""
    frames, resources = thread["frameTable"], thread["resourceTable"]
    names = []
    for func, address in zip(frames["func"], frames["address"], strict=True):
        res = thread["funcTable"]["resource"][func]
        lib = (
            libs[resources["lib"][res]]["debugName"]
            if res is not None and res >= 0
            else "?"
        )
        names.append(symbols.lookup(lib, address))
    return names


def stacks(path: str) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(thread name, frames)`` per sample, frames leaf first."""
    with gzip.open(path) as f:
        profile = json.load(f)
    sidecar = Path(path.removesuffix(".gz")).with_suffix(".json.syms.json")
    symbols = _Symbols(json.loads(sidecar.read_text()))
    for thread in profile["threads"]:
        names = _frame_names(thread, profile["libs"], symbols)
        stack_table = thread["stackTable"]
        for stack in thread["samples"]["stack"]:
            chain = []
            while stack is not None:
                chain.append(names[stack_table["frame"][stack]])
                stack = stack_table["prefix"][stack]
            if chain:
                yield thread["name"], chain


def tsfx_function(chain: list[str]) -> str | None:
    """Return the innermost tsfx feature function on the stack, if any."""
    for frame in chain:
        m = _TSFX_FUNCTION.search(frame)
        if m:
            return m.group(0).removeprefix("tsfx::")
    return None


def _named_leaf(chain: list[str]) -> str:
    """Return the innermost frame with a symbol name.

    glibc ships without symbols, so its internals show up as ``fun_<addr>``
    (or ``<lib+addr>``) below the named function that called them, such as
    ``malloc``.
    """
    for frame in chain:
        if not frame.startswith(("fun_", "<")) or " as " in frame:
            return frame
    return chain[0]


def leaf_kind(chain: list[str]) -> str | None:
    """Classify a sample as idle, copy or alloc time by its innermost frame."""
    leaf = _named_leaf(chain)
    if any(k in leaf for k in IDLE_LEAVES):
        return "idle"
    if any(k in leaf for k in COPY_LEAVES):
        return "copy"
    if any(k in leaf for k in ALLOC_LEAVES) or "libsystem_malloc" in leaf:
        return "alloc"
    return None
