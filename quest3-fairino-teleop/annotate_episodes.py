"""
annotate_episodes.py — label recorded episodes for ACT training.

Sets `instruction` and `success` in each episode's meta.json. Episodes are
recorded with success=null; you confirm/label them here before export.

Usage:
  python3 annotate_episodes.py                       # interactive, all unlabeled
  python3 annotate_episodes.py --dir ./episodes
  python3 annotate_episodes.py --episode 3 --success true --instruction "pick red block"
  python3 annotate_episodes.py --list                # show status table
"""
import argparse
import json
import os

from config import LOG_DIR


def _episodes(root: str):
    out = []
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "meta.json")
        if d.startswith("episode_") and os.path.exists(mp):
            out.append((d, mp))
    return out


def _load(mp):
    with open(mp) as f:
        return json.load(f)


def _save(mp, m):
    with open(mp, "w") as f:
        json.dump(m, f, indent=2)


def show_list(root):
    rows = _episodes(root)
    if not rows:
        print(f"No episodes in {root}")
        return
    print(f"{'episode':<14}{'success':<9}{'steps':<7}{'dur':<7}instruction")
    for d, mp in rows:
        m = _load(mp)
        q = m.get("quality_metrics", {})
        succ = {True: "✓", False: "✗", None: "—"}.get(m.get("success"), "?")
        print(f"{d:<14}{succ:<9}{m.get('num_steps',0):<7}"
              f"{m.get('duration_s',0):<7.1f}{m.get('instruction','') or '(none)'}"
              f"   [g={q.get('grasps')} sm={q.get('smoothness')}]")


def _parse_bool(s):
    return None if s is None else str(s).lower() in ("1", "true", "yes", "y", "ok", "success")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=LOG_DIR)
    ap.add_argument("--episode", type=int, help="episode index to label directly")
    ap.add_argument("--success", help="true/false (non-interactive)")
    ap.add_argument("--instruction", help="instruction text (non-interactive)")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    args = ap.parse_args()

    if args.list:
        show_list(args.dir)
        return

    # Direct, non-interactive labeling of one episode.
    if args.episode is not None and (args.success is not None or args.instruction is not None):
        mp = os.path.join(args.dir, f"episode_{args.episode:03d}", "meta.json")
        m = _load(mp)
        if args.success is not None:
            m["success"] = _parse_bool(args.success)
        if args.instruction is not None:
            m["instruction"] = args.instruction
        _save(mp, m)
        print(f"updated episode_{args.episode:03d}: success={m['success']} "
              f"instruction={m['instruction']!r}")
        return

    # Interactive: walk episodes that still need a success label.
    for d, mp in _episodes(args.dir):
        m = _load(mp)
        if m.get("success") is not None:
            continue
        q = m.get("quality_metrics", {})
        print(f"\n{d}  steps={m.get('num_steps')}  dur={m.get('duration_s')}s  "
              f"grasps={q.get('grasps')} regrasps={q.get('regrasps')} smoothness={q.get('smoothness')}")
        print(f"  instruction: {m.get('instruction') or '(none)'}")
        ans = input("  success? [y/n/s=skip/q=quit]  ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        m["success"] = (ans == "y")
        ins = input("  instruction (enter = keep current): ").strip()
        if ins:
            m["instruction"] = ins
        _save(mp, m)
        print(f"  → success={m['success']} instruction={m['instruction']!r}")

    print("\nDone. Run with --list to review.")


if __name__ == "__main__":
    main()
