"""
Run every exercise stub and report status:
  PASS  — you implemented it and the self-check passed
  TODO  — not implemented yet (raises NotImplementedError)
  FAIL  — implemented but a check failed (shows the error)

    python docs/exercises/check_all.py
    python docs/exercises/check_all.py --solutions   # run the reference solutions instead
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USE_SOLUTIONS = "--solutions" in sys.argv
SRC = HERE / "solutions" if USE_SOLUTIONS else HERE

files = sorted(p for p in SRC.glob("[0-9][0-9]_*.py"))
print(f"running {'solutions' if USE_SOLUTIONS else 'exercises'} in {SRC}\n")

results = {}
for f in files:
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)      # runs the module-level _check() via __main__? no
        mod._check()
        results[f.name] = "PASS"
    except NotImplementedError:
        results[f.name] = "TODO"
    except AssertionError as e:
        results[f.name] = f"FAIL (assert: {e})"
    except Exception as e:
        results[f.name] = f"FAIL ({type(e).__name__}: {e})"

print("\n" + "=" * 50)
for name, status in results.items():
    print(f"  {status.split()[0]:5}  {name}")
done = sum(1 for s in results.values() if s == "PASS")
print(f"\n{done}/{len(results)} passing")
