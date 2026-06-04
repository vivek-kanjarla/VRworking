#!/usr/bin/env python3
"""
tune.py — interactive parameter tuner for the SO-101 → FR5 teleoperation system.

  ↑ ↓        navigate parameters
  ← →        adjust (small step)
  [ ]        adjust (large step)
  Enter      type a value directly

  Tab        cycle to next preset
  S          save current values as a new named preset
  O          overwrite current preset with current values
  D          delete current preset  (cannot delete 'main')
  R          reload current preset  (discard edits)

  A          apply current values → config.py + singularity.py
  Q / Esc    quit
"""

import curses
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

HERE           = os.path.dirname(os.path.abspath(__file__))
PRESETS_FILE   = os.path.join(HERE, "presets.json")
CONFIG_PY      = os.path.join(HERE, "config.py")
SINGULARITY_PY = os.path.join(HERE, "singularity.py")


# ── Parameter model ───────────────────────────────────────────────────────────

@dataclass
class Param:
    key:     str
    default: Any
    dtype:   type
    step:    float
    big:     float
    lo:      float | None = None
    hi:      float | None = None
    fmt:     str = "{:.3f}"
    value:   Any = field(init=False)

    def __post_init__(self):
        self.value = self.default

    def show(self) -> str:
        return str(int(self.value)) if self.dtype is int else self.fmt.format(float(self.value))

    def adjust(self, delta: float):
        v = float(self.value) + delta
        if self.lo is not None: v = max(float(self.lo), v)
        if self.hi is not None: v = min(float(self.hi), v)
        self.value = int(round(v)) if self.dtype is int else self.dtype(v)

    def parse(self, s: str) -> bool:
        try:
            v = self.dtype(s.strip())
            if self.lo is not None: v = self.dtype(max(float(self.lo), float(v)))
            if self.hi is not None: v = self.dtype(min(float(self.hi), float(v)))
            self.value = v
            return True
        except (ValueError, TypeError):
            return False

    def to_json(self):   return self.value
    def from_json(self, v): self.value = self.dtype(v)


@dataclass
class Section:
    label: str


# ── All tunable parameters ────────────────────────────────────────────────────

def build_params() -> list:
    P, S = Param, Section
    return [
        S("CONTROL LOOP"),
        P("LOOP_HZ",                     125,   int,   1,     25,    60,    1000,  "{}"),

        S("JOINT MAPPING  —  scale  (+ same direction  − reverse)"),
        P("JOINT_SCALE[shoulder_pan]",   -1.0,  float, 0.05,  0.5,  -1.0,   1.0),
        P("JOINT_SCALE[shoulder_lift]",   1.0,  float, 0.05,  0.5,  -1.0,   1.0),
        P("JOINT_SCALE[elbow_flex]",      1.0,  float, 0.05,  0.5,  -1.0,   1.0),
        P("JOINT_SCALE[wrist_flex]",      1.0,  float, 0.05,  0.5,  -1.0,   1.0),
        P("JOINT_SCALE[wrist_roll]",      1.0,  float, 0.05,  0.5,  -1.0,   1.0),

        S("JOINT MAPPING  —  amplification  (1.0 = 1 : 1)"),
        P("JOINT_AMP[shoulder_pan]",      1.5,  float, 0.1,   0.5,   0.1,  10.0),
        P("JOINT_AMP[shoulder_lift]",     2.0,  float, 0.1,   0.5,   0.1,  10.0),
        P("JOINT_AMP[elbow_flex]",        2.0,  float, 0.1,   0.5,   0.1,  10.0),
        P("JOINT_AMP[wrist_flex]",        3.0,  float, 0.1,   0.5,   0.1,  10.0),
        P("JOINT_AMP[wrist_roll]",        1.5,  float, 0.1,   0.5,   0.1,  10.0),
        P("FR5_J4_FROZEN_DEG",            0.0,  float, 1.0,  10.0, -170.0, 170.0, "{:.1f}"),

        S("SAFETY  —  rate limits  (deg/cycle  ×  125 Hz  =  deg/s)"),
        P("MAX_DELTA_DEG_PER_CYCLE",      0.04, float, 0.005, 0.02,  0.001,  2.0,  "{:.4f}"),
        P("MAX_DELTA_PER_JOINT[J1]",      0.08, float, 0.005, 0.02,  0.001,  2.0,  "{:.4f}"),
        P("MAX_DELTA_PER_JOINT[J2]",      0.06, float, 0.005, 0.02,  0.001,  2.0,  "{:.4f}"),
        P("MAX_DELTA_PER_JOINT[J3]",      0.06, float, 0.005, 0.02,  0.001,  2.0,  "{:.4f}"),
        P("MAX_DELTA_PER_JOINT[J4]",      0.20, float, 0.005, 0.05,  0.001,  2.0,  "{:.4f}"),
        P("MAX_DELTA_PER_JOINT[J5]",      0.04, float, 0.005, 0.02,  0.001,  2.0,  "{:.4f}"),
        P("MAX_DELTA_PER_JOINT[J6]",      0.10, float, 0.005, 0.02,  0.001,  2.0,  "{:.4f}"),

        S("SAFETY  —  ServoJ parameters"),
        P("FR5_SERVO_VEL",                2,    int,   1,     5,     1,    100,   "{}"),
        P("FR5_FILTER_T",                 0.12, float, 0.01,  0.05,  0.0,   1.0),

        S("JOINT LIMITS  (deg)"),
        P("FR5_JOINT_LIMITS[J1].lo",   -170.0, float, 1.0,  10.0, -360.0,   0.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J1].hi",    170.0, float, 1.0,  10.0,    0.0, 360.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J2].lo",   -260.0, float, 1.0,  10.0, -360.0,   0.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J2].hi",     80.0, float, 1.0,  10.0,    0.0, 360.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J3].lo",   -155.0, float, 1.0,  10.0, -360.0,   0.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J3].hi",    155.0, float, 1.0,  10.0,    0.0, 360.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J4].lo",   -260.0, float, 1.0,  10.0, -360.0,   0.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J4].hi",     80.0, float, 1.0,  10.0,    0.0, 360.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J5].lo",   -170.0, float, 1.0,  10.0, -360.0,   0.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J5].hi",    170.0, float, 1.0,  10.0,    0.0, 360.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J6].lo",   -170.0, float, 1.0,  10.0, -360.0,   0.0,  "{:.1f}"),
        P("FR5_JOINT_LIMITS[J6].hi",    170.0, float, 1.0,  10.0,    0.0, 360.0,  "{:.1f}"),

        S("GRIPPER  (DH AG-160-95)"),
        P("GRIPPER_OPEN_PCT",           100,   int,   5,    20,     0,   100,  "{}"),
        P("GRIPPER_CLOSE_PCT",            0,   int,   5,    20,     0,   100,  "{}"),
        P("GRIPPER_VEL_PCT",             50,   int,   5,    20,     0,   100,  "{}"),
        P("GRIPPER_FORCE_PCT",           50,   int,   5,    20,     0,   100,  "{}"),
        P("GRIPPER_MAXTIME_MS",        5000,   int, 100,   500,   100, 30000, "{}"),

        S("GRIPPER  —  SO-101 normalised thresholds  (0 – 1)"),
        P("SO101_GRIPPER_OPEN_THRESHOLD",  0.65, float, 0.01, 0.05, 0.01, 0.99),
        P("SO101_GRIPPER_CLOSE_THRESHOLD", 0.35, float, 0.01, 0.05, 0.01, 0.99),

        S("SINGULARITY  —  detection thresholds  (deg)"),
        P("WRIST_WARN_DEG",   15.0, float, 0.5, 5.0, 0.0, 90.0, "{:.1f}"),
        P("WRIST_DANGER_DEG",  5.0, float, 0.5, 5.0, 0.0, 90.0, "{:.1f}"),
        P("ELBOW_WARN_DEG",    8.0, float, 0.5, 5.0, 0.0, 90.0, "{:.1f}"),
        P("ELBOW_DANGER_DEG",  3.0, float, 0.5, 5.0, 0.0, 90.0, "{:.1f}"),
    ]


# ── Preset I/O ────────────────────────────────────────────────────────────────

def params_to_dict(items: list) -> dict:
    return {it.key: it.to_json() for it in items if isinstance(it, Param)}

def dict_to_params(items: list, d: dict):
    for it in items:
        if isinstance(it, Param) and it.key in d:
            it.from_json(d[it.key])

def load_presets() -> dict:
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE) as f:
            return json.load(f)
    return {}

def save_presets(presets: dict):
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


# ── Config file writers ───────────────────────────────────────────────────────

def write_config_py(p: dict):
    lo = lambda j: int(p[f"FR5_JOINT_LIMITS[{j}].lo"])
    hi = lambda j: int(p[f"FR5_JOINT_LIMITS[{j}].hi"])

    content = (
        '"""\n'
        'config.py — central configuration for SO-101 → FR5 teleoperation.\n'
        'Edit these values to match your physical setup before running teleop.py.\n'
        '"""\n'
        '\n'
        '# ── SO-101 (leader arm) ───────────────────────────────────────────────────────\n'
        'SO101_PORT     = "/dev/ttyACM0"   # Run: ls /dev/ttyACM* after plugging in USB\n'
        'SO101_BAUDRATE = 1_000_000        # Fixed for Feetech STS3215\n'
        '\n'
        '# Motor IDs on the SO-101 bus (1=shoulder_pan ... 5=wrist_roll, 6=gripper)\n'
        'SO101_MOTORS = {\n'
        '    "shoulder_pan":  1,\n'
        '    "shoulder_lift": 2,\n'
        '    "elbow_flex":    3,\n'
        '    "wrist_flex":    4,\n'
        '    "wrist_roll":    5,\n'
        '}\n'
        'SO101_GRIPPER_ID = 6   # read separately — not part of arm mapping\n'
        '\n'
        '# ── FR5 (follower cobot) ──────────────────────────────────────────────────────\n'
        'FR5_IP = "192.168.58.2"           # Default Fairino controller IP\n'
        '\n'
        '# ── Control loop ──────────────────────────────────────────────────────────────\n'
        f'LOOP_HZ     = {p["LOOP_HZ"]}                 # ServoJ must run between 60–1000 Hz\n'
        f'LOOP_PERIOD = 1.0 / LOOP_HZ      # ~{1.0/p["LOOP_HZ"]:.3f} s between calls\n'
        '\n'
        '# ── Joint mapping: SO-101 (5 joints) → FR5 (6 joints) ────────────────────────\n'
        '#\n'
        '# FR5 J5 has NO SO-101 counterpart — frozen at home position.\n'
        '#\n'
        f'FR5_J4_FROZEN_DEG = {p["FR5_J4_FROZEN_DEG"]:.1f}\n'
        '\n'
        '# Scale: +1.0 = same direction, -1.0 = reversed\n'
        '# Index: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]\n'
        f'JOINT_SCALE = [{p["JOINT_SCALE[shoulder_pan]"]:.3f}, {p["JOINT_SCALE[shoulder_lift]"]:.3f}, '
        f'{p["JOINT_SCALE[elbow_flex]"]:.3f}, {p["JOINT_SCALE[wrist_flex]"]:.3f}, {p["JOINT_SCALE[wrist_roll]"]:.3f}]\n'
        '\n'
        '# Amplification: 1° SO-101 → N° FR5\n'
        '# Index: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]\n'
        f'JOINT_AMP = [{p["JOINT_AMP[shoulder_pan]"]:.2f}, {p["JOINT_AMP[shoulder_lift]"]:.2f}, '
        f'{p["JOINT_AMP[elbow_flex]"]:.2f}, {p["JOINT_AMP[wrist_flex]"]:.2f}, {p["JOINT_AMP[wrist_roll]"]:.2f}]\n'
        '\n'
        '# ── Safety ────────────────────────────────────────────────────────────────────\n'
        f'MAX_DELTA_DEG_PER_CYCLE = {p["MAX_DELTA_DEG_PER_CYCLE"]:.4f}    # 0.04 × 125Hz = 5°/s\n'
        '\n'
        '# Per-joint rate limits (degrees/cycle) for [J1, J2, J3, J4, J5, J6]\n'
        f'MAX_DELTA_PER_JOINT = [{p["MAX_DELTA_PER_JOINT[J1]"]:.4f}, {p["MAX_DELTA_PER_JOINT[J2]"]:.4f}, '
        f'{p["MAX_DELTA_PER_JOINT[J3]"]:.4f}, {p["MAX_DELTA_PER_JOINT[J4]"]:.4f}, '
        f'{p["MAX_DELTA_PER_JOINT[J5]"]:.4f}, {p["MAX_DELTA_PER_JOINT[J6]"]:.4f}]\n'
        '#                       J1     J2     J3     J4    J5    J6\n'
        '\n'
        f'FR5_SERVO_VEL           = {p["FR5_SERVO_VEL"]}       # ServoJ velocity % — start low, tune up\n'
        f'FR5_FILTER_T            = {p["FR5_FILTER_T"]:.2f}    # ServoJ trajectory filter (seconds)\n'
        '\n'
        '# From GetJointSoftLimitDeg() on this controller, with 5° margin inside each limit.\n'
        'FR5_JOINT_LIMITS = [\n'
        f'    ({lo("J1")}, {hi("J1")}),   # J1\n'
        f'    ({lo("J2")}, {hi("J2")}),   # J2\n'
        f'    ({lo("J3")}, {hi("J3")}),   # J3\n'
        f'    ({lo("J4")}, {hi("J4")}),   # J4 — frozen in mapper\n'
        f'    ({lo("J5")}, {hi("J5")}),   # J5\n'
        f'    ({lo("J6")}, {hi("J6")}),   # J6\n'
        ']\n'
        '\n'
        '# ── DH AG-160-95 gripper (flange-mounted, via Fairino SDK) ───────────────────\n'
        'GRIPPER_INDEX     = 1      # gripper number configured on the FR5 controller\n'
        'GRIPPER_TYPE      = 0      # 0=parallel gripper (AG-160-95), 1=rotating\n'
        '\n'
        f'GRIPPER_OPEN_PCT  = {p["GRIPPER_OPEN_PCT"]}    # position % sent when opening  (0–100)\n'
        f'GRIPPER_CLOSE_PCT = {p["GRIPPER_CLOSE_PCT"]}      # position % sent when closing\n'
        f'GRIPPER_VEL_PCT   = {p["GRIPPER_VEL_PCT"]}     # movement speed    (0–100)\n'
        f'GRIPPER_FORCE_PCT = {p["GRIPPER_FORCE_PCT"]}     # grip force        (0–100)\n'
        f'GRIPPER_MAXTIME_MS = {p["GRIPPER_MAXTIME_MS"]}  # max travel time before timeout (ms)\n'
        '\n'
        '# SO-101 gripper motor position normalised to [0,1]:\n'
        '#   above OPEN_THRESHOLD  → send open\n'
        '#   below CLOSE_THRESHOLD → send close\n'
        '#   between               → hold (hysteresis)\n'
        'SO101_GRIPPER_RANGE        = (2028/4096*360, 3236/4096*360)  # (178.0°, 284.3°)\n'
        f'SO101_GRIPPER_OPEN_THRESHOLD  = {p["SO101_GRIPPER_OPEN_THRESHOLD"]:.2f}   # norm >= this → open\n'
        f'SO101_GRIPPER_CLOSE_THRESHOLD = {p["SO101_GRIPPER_CLOSE_THRESHOLD"]:.2f}   # norm <= this → close\n'
        '\n'
        '# ── Data logging ──────────────────────────────────────────────────────────────\n'
        'LOG_DIR = "./episodes"\n'
    )
    with open(CONFIG_PY, "w") as f:
        f.write(content)


def write_singularity_py(p: dict):
    with open(SINGULARITY_PY) as f:
        src = f.read()

    def rep(text, name, val, fmt):
        return re.sub(
            rf'^({re.escape(name)}\s*=\s*)[\d.]+',
            lambda m: m.group(1) + fmt.format(val),
            text, flags=re.MULTILINE,
        )

    src = rep(src, "WRIST_WARN_DEG",   p["WRIST_WARN_DEG"],   "{:.1f}")
    src = rep(src, "WRIST_DANGER_DEG", p["WRIST_DANGER_DEG"], "{:.1f}")
    src = rep(src, "ELBOW_WARN_DEG",   p["ELBOW_WARN_DEG"],   "{:.1f}")
    src = rep(src, "ELBOW_DANGER_DEG", p["ELBOW_DANGER_DEG"], "{:.1f}")

    with open(SINGULARITY_PY, "w") as f:
        f.write(src)


# ── TUI ───────────────────────────────────────────────────────────────────────

HELP = ("↑↓ nav  ←→ step  [] big  Enter edit  "
        "Tab preset  S save  O overwrite  D del  R reload  A apply  Q quit")


class TunerApp:
    def __init__(self):
        self.items   = build_params()
        self.params  = [it for it in self.items if isinstance(it, Param)]
        self.presets = load_presets()

        if "main" not in self.presets:
            self.presets["main"] = params_to_dict(self.params)
            save_presets(self.presets)

        self._refresh_preset_names()
        self.preset_idx = 0
        self.cursor     = 0
        self.scroll     = 0
        self.msg        = ""
        self.dirty      = False

    # ── preset helpers ────────────────────────────────────────────────────────

    def _refresh_preset_names(self):
        self.preset_names = sorted(
            self.presets.keys(), key=lambda k: (k != "main", k)
        )

    @property
    def preset_name(self) -> str:
        return self.preset_names[self.preset_idx]

    def _load_preset(self, name: str):
        dict_to_params(self.items, self.presets[name])
        self.preset_idx = self.preset_names.index(name)
        self.msg   = f"Loaded '{name}'"
        self.dirty = False

    def _save_preset(self, name: str):
        self.presets[name] = params_to_dict(self.params)
        self._refresh_preset_names()
        save_presets(self.presets)
        self.preset_idx = self.preset_names.index(name)
        self.msg   = f"Saved '{name}'"
        self.dirty = False

    def _delete_preset(self):
        name = self.preset_name
        if name == "main":
            self.msg = "Cannot delete 'main'"
            return
        del self.presets[name]
        self._refresh_preset_names()
        save_presets(self.presets)
        self.preset_idx = 0
        self._load_preset(self.preset_names[0])
        self.msg = f"Deleted '{name}'"

    def _apply(self):
        p = params_to_dict(self.params)
        write_config_py(p)
        write_singularity_py(p)
        self.msg = "Applied → config.py + singularity.py"

    # ── navigation ────────────────────────────────────────────────────────────

    def _move(self, delta: int):
        self.cursor = max(0, min(len(self.params) - 1, self.cursor + delta))

    def _cursor_item_idx(self) -> int:
        count = 0
        for i, it in enumerate(self.items):
            if isinstance(it, Param):
                if count == self.cursor:
                    return i
                count += 1
        return 0

    def _ensure_visible(self, height: int):
        ci = self._cursor_item_idx()
        vis = height - 3
        if ci < self.scroll:
            self.scroll = ci
        elif ci - self.scroll >= vis:
            self.scroll = ci - vis + 1

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw(self, stdscr):
        H, W = stdscr.getmaxyx()
        stdscr.erase()
        self._ensure_visible(H)

        # title bar
        dirty_mark = "*" if self.dirty else " "
        title = f" Teleop Tuner  |  preset: [{self.preset_name}]{dirty_mark}"
        stdscr.attron(curses.A_REVERSE)
        stdscr.addstr(0, 0, title.ljust(W - 1))
        stdscr.attroff(curses.A_REVERSE)

        # param rows
        KEY_COL = 4
        VAL_COL = 46
        row = 1
        pcnt = 0
        for idx, it in enumerate(self.items):
            if idx < self.scroll:
                if isinstance(it, Param):
                    pcnt += 1
                continue
            if row >= H - 2:
                break

            if isinstance(it, Section):
                label = f"  {it.label}"
                try:
                    stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                    stdscr.addstr(row, 0, label[:W - 1].ljust(W - 1))
                    stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                except curses.error:
                    pass
                row += 1

            elif isinstance(it, Param):
                selected = (pcnt == self.cursor)
                val_str  = it.show()
                key_str  = it.key

                try:
                    if selected:
                        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                        stdscr.addstr(row, 0, " " * (W - 1))
                        stdscr.addstr(row, KEY_COL, key_str[:VAL_COL - KEY_COL - 2])
                        stdscr.addstr(row, VAL_COL, val_str[:W - VAL_COL - 2])
                        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
                        # bounds hint in dim
                        if it.lo is not None and it.hi is not None:
                            hint = f"  [{it.lo:.3g}…{it.hi:.3g}]"
                            hcol = VAL_COL + len(val_str) + 2
                            if hcol + len(hint) < W - 1:
                                stdscr.attron(curses.A_DIM | curses.color_pair(1))
                                stdscr.addstr(row, hcol, hint)
                                stdscr.attroff(curses.A_DIM | curses.color_pair(1))
                    else:
                        stdscr.addstr(row, KEY_COL, key_str[:VAL_COL - KEY_COL - 2])
                        stdscr.attron(curses.color_pair(3))
                        stdscr.addstr(row, VAL_COL, val_str[:W - VAL_COL - 1])
                        stdscr.attroff(curses.color_pair(3))
                except curses.error:
                    pass

                pcnt += 1
                row  += 1

        # preset list bar
        names_str = "  ".join(
            f"[{n}]" if i == self.preset_idx else n
            for i, n in enumerate(self.preset_names)
        )
        try:
            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(H - 2, 0, f" Presets: {names_str}"[:W - 1].ljust(W - 1))
            stdscr.attroff(curses.color_pair(2))
        except curses.error:
            pass

        # help / status bar
        status = f" {self.msg}" if self.msg else f" {HELP}"
        try:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(H - 1, 0, status[:W - 1].ljust(W - 1))
            stdscr.attroff(curses.A_REVERSE)
        except curses.error:
            pass

    # ── inline text prompt ────────────────────────────────────────────────────

    def _prompt(self, stdscr, label: str) -> str | None:
        H, W = stdscr.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        buf = ""

        while True:
            line = f" {label}: {buf}"
            try:
                stdscr.addstr(H - 1, 0, line[:W - 1].ljust(W - 1), curses.A_REVERSE)
            except curses.error:
                pass
            stdscr.refresh()
            ch = stdscr.getch()

            if ch in (10, 13):
                break
            elif ch == 27:
                buf = None
                break
            elif ch in (127, curses.KEY_BACKSPACE):
                buf = buf[:-1]
            elif 32 <= ch < 127:
                buf += chr(ch)

        curses.noecho()
        curses.curs_set(0)
        return buf

    # ── main loop ─────────────────────────────────────────────────────────────

    def _main(self, stdscr):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)   # selected row
        curses.init_pair(2, curses.COLOR_CYAN,  -1)                  # section / preset bar
        curses.init_pair(3, curses.COLOR_GREEN, -1)                  # values
        curses.curs_set(0)
        stdscr.keypad(True)
        self._load_preset("main")

        while True:
            self._draw(stdscr)
            stdscr.refresh()
            ch   = stdscr.getch()
            param = self.params[self.cursor]

            if ch == curses.KEY_UP:
                self._move(-1);  self.msg = ""
            elif ch == curses.KEY_DOWN:
                self._move(1);   self.msg = ""
            elif ch == curses.KEY_LEFT:
                param.adjust(-param.step);  self.dirty = True;  self.msg = ""
            elif ch == curses.KEY_RIGHT:
                param.adjust(param.step);   self.dirty = True;  self.msg = ""
            elif ch == ord("["):
                param.adjust(-param.big);   self.dirty = True;  self.msg = ""
            elif ch == ord("]"):
                param.adjust(param.big);    self.dirty = True;  self.msg = ""

            elif ch in (10, 13):
                val = self._prompt(stdscr, param.key)
                if val and val.strip():
                    if param.parse(val):
                        self.dirty = True
                        self.msg = f"Set {param.key} = {param.show()}"
                    else:
                        self.msg = f"Invalid: {val!r}"
                else:
                    self.msg = ""

            elif ch == ord("\t"):
                self.preset_idx = (self.preset_idx + 1) % len(self.preset_names)
                self._load_preset(self.preset_names[self.preset_idx])
            elif ch == curses.KEY_BTAB:
                self.preset_idx = (self.preset_idx - 1) % len(self.preset_names)
                self._load_preset(self.preset_names[self.preset_idx])

            elif ch in (ord("s"), ord("S")):
                name = self._prompt(stdscr, "Save preset as")
                if name and name.strip():
                    self._save_preset(name.strip())
                else:
                    self.msg = "Cancelled"

            elif ch in (ord("o"), ord("O")):
                self._save_preset(self.preset_name)

            elif ch in (ord("d"), ord("D")):
                self._delete_preset()

            elif ch in (ord("r"), ord("R")):
                self._load_preset(self.preset_name)

            elif ch in (ord("a"), ord("A")):
                self._apply()

            elif ch in (ord("q"), ord("Q"), 27):
                break

    def run(self):
        curses.wrapper(self._main)


if __name__ == "__main__":
    TunerApp().run()
