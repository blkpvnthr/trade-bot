"""
conviction_portfolio_manager.py

Manages the long-term conviction portfolio — govtech, noveltech, consumerhype positions
held separately from the intraday trading engine. These positions are protected from
intraday liquidation and tracked independently from the active trading universe.

Stdlib only. No pandas, numpy, alpaca, or trading module imports.
"""

import argparse
import json
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConvictionPosition:
    symbol: str
    category: str          # one of: "govtech", "noveltech", "consumerhype"
    thesis: str
    entry_date: str        # ISO date YYYY-MM-DD
    entry_price: float
    shares: int
    notes: str = ""
    thesis_review_date: str = ""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ConvictionPortfolioManager:
    """
    Tracks long-term conviction positions that are isolated from intraday
    trading capital and protected from automatic liquidation.
    """

    CONVICTION_CAPITAL_PCT: float = 0.25

    VALID_CATEGORIES = {"govtech", "noveltech", "consumerhype"}

    def __init__(self, state_file: Path) -> None:
        self._state_file = state_file
        self._positions: list[ConvictionPosition] = []
        # load_warnings is populated by _load(); callers (e.g. trade.py) can
        # read it after construction to emit JSONL events for any parse errors.
        self.load_warnings: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Read JSON state file; start empty if the file does not exist.

        Unknown fields in stored positions are silently stripped for forward
        compatibility (schema evolution). Only the individual bad record is
        skipped — not the entire portfolio. Parse errors are collected in
        self.load_warnings so callers can log them through the system's JSONL
        event infrastructure without requiring a trading-module import here.
        """
        self.load_warnings.clear()
        if not self._state_file.exists():
            self._positions = []
            return

        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            msg = f"could not read state file ({exc}); starting empty"
            print(f"[conviction] Warning: {msg}")
            self.load_warnings.append(msg)
            self._positions = []
            return

        # Strip unknown fields so extra keys added by future versions of this
        # file do not cause TypeError on older versions.
        _known = {f.name for f in fields(ConvictionPosition)}
        self._positions = []
        for p in raw.get("positions", []):
            if not isinstance(p, dict):
                msg = f"skipping non-dict position record (got {type(p).__name__})"
                print(f"[conviction] Warning: {msg}")
                self.load_warnings.append(msg)
                continue
            filtered = {k: v for k, v in p.items() if k in _known}
            try:
                self._positions.append(ConvictionPosition(**filtered))
            except (TypeError, ValueError, KeyError) as exc:
                sym = p.get("symbol", "?")
                msg = f"skipping malformed position '{sym}': {exc}"
                print(f"[conviction] Warning: {msg}")
                self.load_warnings.append(msg)

    def _save(self) -> None:
        """
        Persist current positions atomically.

        Writes to a sibling .tmp file first, then renames it over the target.
        Path.replace() maps to os.replace() which is atomic on POSIX — the
        original file stays valid until the instant of swap, so a process kill
        mid-write cannot corrupt the state file.
        """
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "positions": [asdict(p) for p in self._positions],
        }
        tmp = self._state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._state_file)   # atomic on Linux / macOS

    def reload(self) -> None:
        """Re-read the state file from disk. Call to pick up CLI changes made mid-session."""
        self._load()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        thesis: str,
        category: str,
        notes: str = "",
        review_date: str = "",
    ) -> "ConvictionPosition":
        """Add a new conviction position and persist atomically."""
        symbol = symbol.upper()

        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: "
                + ", ".join(sorted(self.VALID_CATEGORIES))
            )

        shares_int = int(shares)
        price_float = float(price)

        if shares_int <= 0:
            raise ValueError(f"shares must be a positive integer, got {shares_int}")
        if price_float <= 0.0:
            raise ValueError(f"price must be positive, got {price_float}")

        if any(p.symbol == symbol for p in self._positions):
            raise ValueError(
                f"Symbol '{symbol}' already exists in the conviction portfolio. "
                "Remove it first before re-adding."
            )

        position = ConvictionPosition(
            symbol=symbol,
            category=category,
            thesis=thesis,
            entry_date=date.today().isoformat(),
            entry_price=price_float,
            shares=shares_int,
            notes=notes,
            thesis_review_date=review_date,
        )
        self._positions.append(position)
        try:
            self._save()
        except Exception:
            # Rollback the in-memory append so memory and disk stay consistent.
            self._positions.pop()
            raise
        return position

    def remove_position(self, symbol: str) -> bool:
        """Remove a position by symbol (case-insensitive). Returns True if found."""
        upper = symbol.upper()
        before = len(self._positions)
        self._positions = [p for p in self._positions if p.symbol.upper() != upper]
        removed = len(self._positions) < before
        if removed:
            self._save()
        return removed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_protected_symbols(self) -> set[str]:
        """Return the set of symbols that must not be auto-liquidated."""
        return {p.symbol.upper() for p in self._positions}

    def get_allocated_capital(self) -> float:
        """Return total capital currently allocated to conviction positions."""
        return sum(p.entry_price * p.shares for p in self._positions)

    def get_available_conviction_budget(self, account_equity: float) -> float:
        """
        Return remaining conviction budget given current account equity.
        Budget = equity * CONVICTION_CAPITAL_PCT - already_allocated.
        May be negative when allocated capital exceeds the target percentage.
        """
        return (account_equity * self.CONVICTION_CAPITAL_PCT) - self.get_allocated_capital()

    def get_summary(self) -> dict:
        """Return a summary dict suitable for logging / display."""
        categories: dict[str, int] = {}
        for p in self._positions:
            categories[p.category] = categories.get(p.category, 0) + 1

        return {
            "count": len(self._positions),
            "symbols": [p.symbol for p in self._positions],
            "allocated_capital": self.get_allocated_capital(),
            "categories": categories,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_STATE = Path(__file__).parent / "system-data" / "conviction_portfolio.json"


def _cmd_list(manager: ConvictionPortfolioManager, _args) -> None:
    positions = list(manager._positions)
    if not positions:
        print("No conviction positions on file.")
        return

    summary = manager.get_summary()
    header = (
        f"{'SYMBOL':<8} {'CATEGORY':<14} {'SHARES':>6} {'ENTRY':>8} "
        f"{'VALUE':>10} {'ENTRY DATE':<12} {'REVIEW DATE':<12} THESIS"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for p in positions:
        value = p.entry_price * p.shares
        print(
            f"{p.symbol:<8} {p.category:<14} {p.shares:>6} {p.entry_price:>8.2f} "
            f"{value:>10.2f} {p.entry_date:<12} {p.thesis_review_date or 'N/A':<12} "
            f"{p.thesis[:60]}"
        )
    print(sep)
    print(
        f"Total positions: {summary['count']}  |  "
        f"Allocated capital: ${summary['allocated_capital']:,.2f}"
    )


def _cmd_add(manager: ConvictionPortfolioManager, args) -> None:
    try:
        position = manager.add_position(
            symbol=args.symbol,
            shares=args.shares,
            price=args.price,
            thesis=args.thesis,
            category=args.category,
            notes=args.notes or "",
            review_date=args.review or "",
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    value = position.entry_price * position.shares
    print(
        f"Added conviction position: {position.symbol} "
        f"({position.shares} shares @ ${position.entry_price:.2f} = ${value:,.2f}) "
        f"[{position.category}]"
    )


def _cmd_remove(manager: ConvictionPortfolioManager, args) -> None:
    removed = manager.remove_position(args.symbol)
    if removed:
        print(f"Removed conviction position: {args.symbol.upper()}")
    else:
        print(f"Symbol not found in conviction portfolio: {args.symbol.upper()}")
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conviction_portfolio_manager",
        description="Manage long-term conviction positions.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=_DEFAULT_STATE,
        help=f"Path to JSON state file (default: {_DEFAULT_STATE})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="Display all conviction positions.")

    # add
    add_p = sub.add_parser("add", help="Add a new conviction position.")
    add_p.add_argument("symbol", metavar="SYMBOL", help="Ticker symbol")
    add_p.add_argument("shares", metavar="SHARES", type=int, help="Number of shares")
    add_p.add_argument("price", metavar="PRICE", type=float, help="Entry price per share")
    add_p.add_argument("--thesis", required=True, help="Investment thesis narrative")
    add_p.add_argument(
        "--category",
        required=True,
        choices=["govtech", "noveltech", "consumerhype"],
        help="Position category",
    )
    add_p.add_argument("--notes", default="", help="Optional notes")
    add_p.add_argument("--review", default="", metavar="DATE", help="Thesis review date (YYYY-MM-DD)")

    # remove
    rm_p = sub.add_parser("remove", help="Remove a conviction position by symbol.")
    rm_p.add_argument("symbol", metavar="SYMBOL", help="Ticker symbol to remove")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    manager = ConvictionPortfolioManager(state_file=args.state_file)

    dispatch = {
        "list": _cmd_list,
        "add": _cmd_add,
        "remove": _cmd_remove,
    }
    dispatch[args.command](manager, args)


if __name__ == "__main__":
    main()
