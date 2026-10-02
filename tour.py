"""The guided tour: checks your work step by step and tells you where you are.

    uv run tour.py      checks the steps in order and stops at the first one that isn't done
    uv run tour.py 4    checks step 4 only
"""

import contextlib
import importlib
import io
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    # (checks module, what you write, where, lesson)
    ("test_01_linear_forward", "Linear: __init__, forward", "backprop/layers.py", "lessons/01-linear-forward.md"),
    ("test_02_linear_backward", "Linear.backward", "backprop/layers.py", "lessons/02-linear-backward.md"),
    ("test_03_sigmoid_forward", "Sigmoid: __init__, forward", "backprop/layers.py", "lessons/03-sigmoid-forward.md"),
    ("test_04_sigmoid_backward", "Sigmoid.backward", "backprop/layers.py", "lessons/04-sigmoid-backward.md"),
    ("test_05_mse_forward", "MSE: __init__, forward", "backprop/losses.py", "lessons/05-mse-forward.md"),
    ("test_06_mse_backward", "MSE.backward", "backprop/losses.py", "lessons/06-mse-backward.md"),
    ("test_07_sequential_forward", "Sequential: __init__, forward", "backprop/network.py", "lessons/07-sequential-forward.md"),
    ("test_08_sequential_backward", "Sequential.backward", "backprop/network.py", "lessons/08-sequential-backward.md"),
    ("test_09_sequential_step", "Sequential.step", "backprop/network.py", "lessons/09-sequential-step.md"),
    ("test_10_train_xor", "train it on XOR", "nothing to write", "lessons/10-train-xor.md"),
]

# Never crash on a console that can't print ✓ or σ: print a ? instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if os.name == "nt":
    os.system("")  # turns on colours in the classic Windows console

COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def paint(text, color):
    codes = {"green": "32", "red": "31", "grey": "90", "bold": "1", "yellow": "33"}
    return f"\033[{codes[color]}m{text}\033[0m" if COLOR else text


def indent(text, prefix="      "):
    return "\n".join(prefix + line for line in str(text).splitlines())


def run_step(module_name):
    """Runs the checks of one step in order.

    Returns (passed checks, (failed check, error, what your code printed) or None).
    """
    module = importlib.import_module(f"checks.{module_name}")
    checks = [f for name, f in vars(module).items() if name.startswith("test_") and callable(f)]
    passed = []
    for check in checks:
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                check()
        except (Exception, SystemExit) as error:
            return passed, (check, error, printed.getvalue())
        passed.append(check)
    return passed, None


def your_frame(error):
    """The innermost frame of the traceback that is in your code, if any."""
    from checks.helpers import IMPL

    yours = (ROOT / IMPL).resolve()
    found = None
    tb = error.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve().is_relative_to(yours):
            found = tb
        tb = tb.tb_next
    return found


def shown_path(filename):
    path = Path(filename).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def where_in_your_code(error):
    """The line of your code where the error happened, if it happened in your code."""
    if isinstance(error, SyntaxError) and error.filename:
        return f"In {shown_path(error.filename)}, line {error.lineno}:\n    {(error.text or '').strip()}"
    tb = your_frame(error)
    if tb is None:
        return None
    summary = traceback.extract_tb(tb, limit=1)[0]
    return f"In {shown_path(summary.filename)}, line {summary.lineno}, in {summary.name}():\n    {summary.line}"


def explain(error):
    """Plain-words help for the errors you're most likely to hit."""
    message = str(error)
    tb = your_frame(error)
    owner = type(tb.tb_frame.f_locals.get("self")).__name__ if tb else None

    if isinstance(error, ValueError) and ("matmul" in message or "not aligned" in message):
        if owner in ("Sigmoid", "MSE"):
            return (
                f"{owner} works entry by entry: nothing is summed over an index, so there's no matrix product.\n"
                "Use element-wise operations: * instead of @."
            )
        return (
            "The shapes don't line up for a matrix product. In A @ B the inner dims must match:\n"
            "(a, k) @ (k, b) gives (a, b), and k disappears. Print both shapes, e.g. print(A.shape, B.shape),\n"
            "write down the shape you want back, and see which one needs a .T"
        )
    if isinstance(error, ValueError) and "broadcast" in message and owner == "Linear":
        return (
            "z must come out (batch_size, n_out): one row per example, one column per neuron.\n"
            "Arrange the product so it has that shape, then a plain + b adds b to every row. Don't reshape b."
        )
    if isinstance(error, ValueError) and "broadcast" in message:
        return (
            "Two arrays with incompatible shapes met in an element-wise operation (+, -, *, /).\n"
            "Element-wise operations need the same shape (or a shape numpy can stretch, like (n,) on (batch, n)).\n"
            "Print both shapes to see which one is off."
        )
    if isinstance(error, TypeError) and "converted to Python scalars" in message:
        return (
            "A function from Python's math module (like math.exp) works on one number at a time, "
            "and you gave it a whole array.\nUse the numpy version, np.exp, which works on every entry at once."
        )
    if isinstance(error, TypeError) and "NoneType" in message and "+=" in message:
        return "You used += on something that is still None. Assign it with = : each call computes it from scratch."
    if "NoneType" in message:
        return (
            "Something is None: a function that doesn't return its result, "
            "or an attribute that was never set to a real value."
        )
    if isinstance(error, NameError) and tb and tb.tb_frame.f_code.co_name == "backward":
        return (
            "Variables from forward don't exist in backward: each call has its own local variables.\n"
            "If backward needs something that only forward sees, where could forward keep it?"
        )
    if isinstance(error, ModuleNotFoundError) and error.name in ("layers", "losses", "network"):
        return f"Inside the backprop package, import with the package name: from backprop.{error.name} import ..."
    if isinstance(error, NameError) and error.name in ("Linear", "Sigmoid", "MSE"):
        return (
            f"{error.name} isn't imported in this file. Add from backprop.layers import {error.name}"
            if error.name != "MSE" else "MSE isn't imported in this file. Add from backprop.losses import MSE"
        ) + " (or, for step 9, check hasattr(layer, \"W\") instead)."
    if isinstance(error, AttributeError) and "'Sigmoid' object has no attribute" in message:
        return "Not every layer has parameters: Sigmoid has no W, b, dW or db. Only update the layers that have them."
    if isinstance(error, AttributeError):
        return ("Check the spelling, and that the attribute was set before this line runs "
                "(the names the checks look for are W, b, dW and db).")
    return None


def describe(number, error):
    if isinstance(error, NotImplementedError):
        needed = re.match(r"Step (\d+)", str(error))
        if needed and int(needed.group(1)) != number:
            return f"This check needs step {needed.group(1)}, which isn't written yet ({error})."
        return f"Not written yet ({error}).\nReplace the `raise NotImplementedError(...)` line with your code."
    if isinstance(error, AssertionError):
        return str(error)
    if isinstance(error, SystemExit):
        return "Your code called exit() (or quit(), or sys.exit()), which stopped the check.\n" \
               "Remove it: print() is enough to look at values."
    parts = [where_in_your_code(error), f"{type(error).__name__}: {error}", explain(error)]
    return "\n\n".join(part for part in parts if part)


def report(number, step, passed, failure):
    _, title, file, lesson = step
    print(paint(f"Step {number} of {len(STEPS)}: {title}", "bold"))
    print(f"  read   {lesson}")
    if file != "nothing to write":
        if ": " in title:
            owner, methods = title.split(": ")
            title = " and ".join(f"{owner}.{method}" for method in methods.split(", "))
        print(f"  write  {title} in {file}")
    print()
    for check in passed:
        print(paint(f"  ✓ {check.__doc__}", "green"))
    if failure:
        check, error, printed = failure
        print(paint(f"  ✗ {check.__doc__}", "red"))
        print()
        print(indent(describe(number, error)))
        if printed:
            lines = printed.splitlines()
            shown = f" (the first 20 of {len(lines)} lines)" if len(lines) > 20 else ""
            print()
            print(indent(f"Your code printed this while the check ran{shown}:"))
            print(indent("\n".join(lines[:20]), "        "))
    print()


def load_your_files():
    """Imports your three files first, so a broken file is reported as such."""
    from checks.helpers import IMPL

    for name in ("layers", "losses", "network"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(f"{IMPL}.{name}")
        except (Exception, SystemExit) as error:
            print(paint(f"{IMPL}/{name}.py can't be loaded, so nothing in it can be checked yet:", "red"))
            print()
            print(indent(describe(0, error)))
            print()
            return False
    return True


def board(done, current):
    print()
    print(paint(f"Backprop from scratch: {done} of {len(STEPS)} steps done", "bold"))
    print()
    for number, (_, title, file, _) in enumerate(STEPS, 1):
        line = f"{number:>2}  {title:<31}{file}"
        if number <= done:
            print(paint(f"  ✓ {line}", "green"))
        elif current and number == current[0]:
            print(paint(f"  ✗ {line}", "red") + paint("   <- you are here", "yellow"))
        else:
            print(paint(f"  · {line}", "grey"))
    print()


def main(args):
    if args and (args[0] in ("-h", "--help") or not args[0].isdigit() or not 1 <= int(args[0]) <= len(STEPS)):
        print(__doc__.strip())
        print(f"\nThere are steps 1 to {len(STEPS)}.")
        return 0 if args[0] in ("-h", "--help") else 2

    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy isn't installed. Run the tour with uv, which installs it for you:\n\n    uv run tour.py")
        return 1

    if not load_your_files():
        return 1

    if args:
        number = int(args[0])
        passed, failure = run_step(STEPS[number - 1][0])
        report(number, STEPS[number - 1], passed, failure)
        if not failure:
            print(paint("  This step is done.", "green"))
        return 1 if failure else 0

    done, current = 0, None
    for number, step in enumerate(STEPS, 1):
        if sys.stdout.isatty():
            print(paint(f"  checking step {number}...", "grey"), end="\r", flush=True)
        passed, failure = run_step(step[0])
        if failure:
            current = (number, step, passed, failure)
            break
        done = number
    if sys.stdout.isatty():
        print(" " * 30, end="\r")

    board(done, current)
    if current:
        report(*current)
        print(paint("Run `uv run tour.py` again whenever you change something.", "grey"))
        return 1

    print(paint("You wrote backpropagation from scratch. Watch your network learn:", "green"))
    print()
    print("    uv run train_xor.py")
    print()
    print("Then lessons/10-train-xor.md has a few experiments to try.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n\nStopped. If the tour seemed stuck, look for a loop in your code that never ends.")
        sys.exit(130)
