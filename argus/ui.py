"""Colorful, compact terminal UI and ASCII animations for Linux Face Unlock.

Provides lightweight, zero-dependency visual feedback:
- Compact animated reticle during scanning
- Non-technical, clean success checkmarks
- Informative, clear failure banners
- Step-by-step enrollment progress animations
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional


# ANSI color codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[1;32m"
C_CYAN = "\033[1;36m"
C_MAGENTA = "\033[1;35m"
C_YELLOW = "\033[1;33m"
C_RED = "\033[1;31m"
C_BLUE = "\033[1;34m"
C_WHITE = "\033[1;37m"

CLEAR_LINE = "\r\033[K"


class BiometricSpinner:
    """Compact animated biometric scanner reticle for TTY output."""

    FRAMES = [
        (C_CYAN, "◖ ◉ ◗"),
        (C_BLUE, "◖ ◈ ◗"),
        (C_MAGENTA, "◖ ● ◗"),
        (C_YELLOW, "◖ ◎ ◗"),
        (C_CYAN, "◖ ◐ ◗"),
        (C_BLUE, "◖ ◑ ◗"),
    ]

    def __init__(self, message: str = "Scanning face or touch sensor...", stream=sys.stderr):
        self.message = message
        self.stream = stream
        self.index = 0
        self.is_tty = hasattr(stream, "isatty") and stream.isatty()

    def tick(self) -> None:
        """Render the next animation frame."""
        if not self.is_tty:
            return
        color, symbol = self.FRAMES[self.index % len(self.FRAMES)]
        self.index += 1
        output = f"{CLEAR_LINE}{C_BOLD}[ {color}{symbol}{C_RESET}{C_BOLD} ]{C_RESET} {C_WHITE}{self.message}{C_RESET}"
        self.stream.write(output)
        self.stream.flush()

    def clear(self) -> None:
        """Clear current animation line."""
        if self.is_tty:
            self.stream.write(CLEAR_LINE)
            self.stream.flush()

    def success(self, message: str = "Access Granted") -> None:
        """Render success banner without technical metrics."""
        if self.is_tty:
            self.stream.write(f"{CLEAR_LINE}{C_GREEN}✔ [{message}]{C_RESET}\n")
        else:
            self.stream.write(f"[FaceUnlock] ✔ {message}\n")
        self.stream.flush()

    def failure(self, reason: str = "Authentication failed") -> None:
        """Render clear, non-technical failure reason."""
        if self.is_tty:
            self.stream.write(f"{CLEAR_LINE}{C_YELLOW}✗ [{reason}]{C_RESET}\n")
        else:
            self.stream.write(f"[FaceUnlock] ✗ {reason}\n")
        self.stream.flush()


def render_banner(title: str, subtitle: Optional[str] = None) -> None:
    """Render a modern ASCII banner header."""
    width = 54
    print(f"\n{C_CYAN}╭{'─' * width}╮{C_RESET}")
    title_pad = (width - len(title) - 2) // 2
    right_pad = width - len(title) - 2 - title_pad
    print(f"{C_CYAN}│{C_RESET}{' ' * title_pad}{C_BOLD}{C_WHITE}{title}{C_RESET}{' ' * right_pad}{C_CYAN}│{C_RESET}")
    if subtitle:
        sub_pad = (width - len(subtitle) - 2) // 2
        sub_rpad = width - len(subtitle) - 2 - sub_pad
        print(f"{C_CYAN}│{C_RESET}{' ' * sub_pad}{C_DIM}{subtitle}{C_RESET}{' ' * sub_rpad}{C_CYAN}│{C_RESET}")
    print(f"{C_CYAN}╰{'─' * width}╯{C_RESET}\n")


def render_step_progress(step: int, total: int, instruction: str) -> None:
    """Render a colorful step progress indicator during enrollment."""
    filled = "●" * step
    empty = "○" * (total - step)
    dots = f"{C_CYAN}{filled}{C_DIM}{empty}{C_RESET}"
    print(f"{C_BOLD}[{dots}{C_BOLD}] Step {step}/{total}:{C_RESET} {C_WHITE}{instruction}{C_RESET}")


def animate_countdown(seconds: int = 2, message: str = "Position face") -> None:
    """Render a smooth countdown animation with camera icon."""
    if not sys.stdout.isatty():
        time.sleep(seconds)
        return

    frames = ["◖ ◉ ◗", "◖ ◈ ◗", "◖ ● ◗", "◖ ◎ ◗"]
    steps = int(seconds * 10)
    for i in range(steps):
        remaining = max(1, seconds - int(i / 10))
        f = frames[i % len(frames)]
        sys.stdout.write(f"{CLEAR_LINE}  {C_CYAN}[ {f} ]{C_RESET} {message}... {C_BOLD}{remaining}s{C_RESET}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f"{CLEAR_LINE}  {C_GREEN}📷 [ SNAP! ]{C_RESET}\n")
    sys.stdout.flush()
