# Clean Python: Architecture, Workflow & Code Quality
### A self-paced study guide in ten units, built around your own robot: the JETANK garden guard
*Version 3.1 (2026-11-05). Replaces pluggy with Jetank as the main codebase, and adds Robert C. Martin's* Clean Code *(2nd edition, 2025) as the second companion book. 3.1: units instead of weeks, chapter map for the 2nd edition, a core path per unit. The guide lives in `Jetank/resources/`; version 2 (pluggy-based) and version 1 are kept in `~/Projects/_old/`.*

> **Who this is for:** an experienced engineer (hardware/ASIC/signal processing) who knows OOP concepts and wants to write Python the way well-run projects do: clean structure, explicit interfaces, solid data models, and a professional tool chain.
>
> **Pace:** self-paced. A unit is roughly "one week of 4–6 hours" worth of material, but there is no calendar. With limited time, expect a unit to take two or three weeks, and that's fine. Each unit has a **core path** (marked at the top): do that and the unit is done. Everything else is extra.
>
> **Each unit has five parts:** **Concepts** (the theory), **Clean Code** (the chapters of Martin's book that match), **Jetank lens** (the same idea in your own code), **Exercises** (most of them build the next real feature), and a small **Neovim step** (Appendix E).
>
> **Main codebase:** `~/Projects/Jetank`, the `src/jetank` package (~4,000 lines of source, ~460 unit tests, strict mypy). Over the ten units you build the **garden guard**: camera → detector → tracker → behaviour → drive.

---

## Table of contents

- [Unit 0 – How to use this guide](#unit-0--how-to-use-this-guide)
- [Unit 1 – Orientation: environment, layout and the map of Jetank](#unit-1--orientation-environment-layout-and-the-map-of-jetank)
- [Unit 2 – Quality gates: pre-commit, CI and an enforced architecture](#unit-2--quality-gates-pre-commit-ci-and-an-enforced-architecture)
- [Unit 3 – Names, functions, comments: refactoring `cli.py`](#unit-3--names-functions-comments-refactoring-clipy)
- [Unit 4 – Objects, data and dependency inversion: designing the tracker](#unit-4--objects-data-and-dependency-inversion-designing-the-tracker)
- [Unit 5 – Testing disciplines and error handling: building the tracker with TDD](#unit-5--testing-disciplines-and-error-handling-building-the-tracker-with-tdd)
- [Unit 6 – Boundaries: the YOLO decoder and the TensorRT adapter](#unit-6--boundaries-the-yolo-decoder-and-the-tensorrt-adapter)
- [Unit 7 – Behaviour and concurrency: the guard state machine](#unit-7--behaviour-and-concurrency-the-guard-state-machine)
- [Unit 8 – Architecture: wiring `jetank guard` end to end](#unit-8--architecture-wiring-jetank-guard-end-to-end)
- [Unit 9 – Continuous design: harden, field-test, document](#unit-9--continuous-design-harden-field-test-document)
- [Unit 10 – Relentless improvement and your own standards](#unit-10--relentless-improvement-and-your-own-standards)
- [Appendix A – Clean code review checklist (Jetank edition)](#appendix-a--clean-code-review-checklist-jetank-edition)
- [Appendix B – Glossary](#appendix-b--glossary)
- [Appendix C – Resource library](#appendix-c--resource-library)
- [Appendix D – Working with Claude without outsourcing the learning](#appendix-d--working-with-claude-without-outsourcing-the-learning)
- [Appendix E – The Neovim track](#appendix-e--the-neovim-track)
- [Appendix F – *Clean Code* in Python: what to adopt, what to adapt](#appendix-f--clean-code-in-python-what-to-adopt-what-to-adapt)

---

## Unit 0 – How to use this guide

### Why the switch from pluggy

Pluggy was chosen as a *model* codebase: small, mature, well tested. Its weakness for you was motivation: you would have learned from code you have no use for. Jetank turns this around. You care whether the robot chases the cat, and that frees attention for the design questions.

There is a second, less obvious reason. **You did not write most of Jetank yourself.** All twenty commits since the scaffold were made together with Claude. For learning, that makes Jetank as foreign as pluggy at first: you have to *read* it to understand it. The difference is that from now on you also *extend* it, and every design decision you make lands on a real robot.

What you lose with pluggy: a codebase reviewed by many experienced maintainers, and a real open-source contribution. Jetank compensates with Cosmic Python (its architecture is exactly the one Jetank uses) and *Clean Code* as outside references. Pluggy remains an optional side track (unit 10).

### Learning principles

1. **Active over passive.** Every concept comes with something to *type, break, or build*. Most exercises build the next real feature of the garden guard.
2. **You write, Claude reviews, by default.** If Claude writes the tracker, you will have a tracker but not the skill. Each exercise carries a mode tag: **[you]** you write it, **[pair]** you write tests or design and Claude fills in, **[review]** Claude writes and you review. The tag is a default, not a law. When time is short and the robot needs to move forward, switch an exercise to **[review]** on purpose and note it in the journal. A conscious shortcut is fine; drifting into "Claude writes everything" is what to avoid. See Appendix D.
3. **Interleave theory and real code.** Each concept appears in the abstract, in a book (Cosmic Python, *Clean Code*), and in Jetank.
4. **Explain to learn.** At the end of each unit (or each sitting, if a unit stretches over weeks), write a few lines in your own words in `resources/JOURNAL.md` (next to this guide; it has templates for sittings, unit summaries, shortcuts and disagreements). If you can't explain it, you haven't learned it yet. The journal also tells you where you stopped when you come back after a break.
5. **Read critically.** *Clean Code* is written with Java examples, and Martin has strong opinions. Most of the advice transfers, some needs adapting, and part of the learning is telling the difference. Appendix F lists the cases, and the book's own appendix, "The Clean Code Debate", shows Martin arguing them out with John Ousterhout.
6. **Change one habit at a time.** The Neovim track adds one small habit per unit. VS Code stays an allowed fallback.
7. **Safety is part of clean code here.** A bug in a notebook gives a wrong number. A bug in Jetank drives the robot into the pond. Bounded loops, safe-state exits and validated limits are design requirements, not afterthoughts.

### Working with interruptions

You'll often have an hour, not an afternoon. The guide is built for that:
- **Small exercises.** Most exercises fit one sitting. The big ones (refactor `cli.py`, build the tracker) are split into numbered steps, and each step ends with green tests and a commit, so you can stop anywhere.
- **Leave a note for future you.** Before stopping, update the "Where I stopped" box at the top of the journal: "next: step 4, test for dropped tracks". It saves the 20 minutes of "where was I?".
- **Reading is a good fit for short slots.** Read the *Clean Code* chapters on the sofa; save keyboard time for exercises.
- **Skip freely outside the core path.** Coming back to an extra exercise months later is better than stalling on it now.

### The roadmap you build along the way

| Unit | Feature you build | Layer |
|---|---|---|
| 1–2 | Tool chain: pre-commit, CI, an import contract | repo |
| 3 | `cli.py` split into small command modules | app |
| 4 | Tracker design: types and interface, no logic yet | domain, interfaces |
| 5 | Tracker implementation with TDD | domain or control |
| 6 | YOLO output decoding + NMS, TensorRT engine adapter, `YoloDetector` | perception, drivers |
| 7 | Guard behaviour state machine (watch → confirm → approach → retreat → cooldown) | control |
| 8 | `jetank guard` wired end to end, first in `--sim` | app |
| 9 | Event recording, field test, docs | drivers, app |
| 10 | Whole-codebase review, `STANDARDS.md` | everything |

### Reading plan: *Clean Code*, 2nd edition, by unit

The 2nd edition (Addison-Wesley, 2025) has 37 chapters in four parts (Code, Design, Architecture, Craftsmanship) and one appendix. Chapters in *italics* are optional.

| Unit | Chapters | Topic |
|---|---|---|
| 1 | 1, 2, *17* | Clean Code, Clean That Code!, *AIs, LLMs, and God Knows What* |
| 2 | 6, 31 | Formatting, Small Cycles (CI, continuous build) |
| 3 | 3, 4, 5, 7, 8, 10, *9*, *11* | First Principles, Meaningful Names, Comments, Clean Functions, Function Heuristics, One Thing, *The Clean Method*, *Be Polite* |
| 4 | 12, 13, 19 | Objects and Data Structures, Clean Classes, The SOLID Principles |
| 5 | 14, 15 | Testing Disciplines, Clean Tests |
| 6 | 26, 18 | Clean Boundaries, Simple Design |
| 7 | 22 | Concurrency |
| 8 | 25, 27, *23*, *24*, *20* | Architectural Boundaries, The Clean Architecture, *The Two Values of Software*, *Independence*, *Component Principles* |
| 9 | 21, 16, 28 | Continuous Design, Acceptance Testing, Harm |
| 10 | 32, Appendix, *29*, *30* | Relentless Improvement, The Clean Code Debate, *No Defect in Behavior or Structure*, *Repeatable Proof* |

Unit 3 is the heaviest reading block, because the 2nd edition spreads functions over five chapters. Chapters 33–37 (productivity, teamwork, estimating, respect, learning) aren't tied to code; read them whenever you like.

Chapter 17 is worth reading early: it's Martin's view on programming with AI assistants, and you'll be working with one throughout. Compare it with the working agreement in Appendix D.

### One-time setup

```text
~/Projects/
├── Jetank/               # the main codebase (git: kim74/jetank)
│   └── resources/        # committed to the repo
│       ├── python-clean-architecture-guide.md   # this guide
│       └── JOURNAL.md    # your notes: one entry per sitting (Python AND Neovim)
├── _old/                 # guide versions 1 and 2 (pluggy-based)
├── cosmicpython_book/    # companion book sources
├── cosmicpython_code/    # companion code, one branch per chapter
└── pluggy/               # optional side track (unit 10)
```

Code runs in two places: unit tests and simulation on the laptop, hardware on the robot (`ssh jetank`). Keep that split in mind. Everything except hardware tests must work on the laptop.

---

## Unit 1 – Orientation: environment, layout and the map of Jetank

> **Neovim step:** Open, move, save, quit. Find files with `<leader><space>`. (Appendix E, step 1.)

### Goals
- Understand Jetank's tool chain and packaging decisions.
- Build a mental map of the package: its layers, its interfaces, its composition root.
- Read *Clean Code* chapters 1 and 2.

> **Core path:** read chapters 1–2 and CLAUDE.md, then exercises 2 and 3.

### Concepts

**1. Virtual environments.** A virtual environment is a private `site-packages` tied to one interpreter. Jetank's laptop `.venv` was created by **uv** with CPython 3.10 (see `.venv/pyvenv.cfg`). On the robot the venv must also see JetPack's system OpenCV (built with GStreamer), which is installed with apt, not pip. Two environments, one `pyproject.toml`.

**2. Declarative metadata: `pyproject.toml`.** One file declares the build backend, the project and its dependencies, optional extras, and the configuration for ruff, mypy, pytest and coverage.

**3. The src layout.** With `src/jetank/`, Python cannot import the package from the working directory by accident. Tests run against the *installed* package.

**4. Pinning the interpreter.** `requires-python = ">=3.10,<3.11"` is unusual. Most libraries support a range of versions. Jetank is an *application* for one platform (JetPack 6, Ubuntu 22.04, Python 3.10), so pinning is honest. It also explains why the code has `class Response(str, Enum)` instead of `StrEnum` (3.11+).

### *Clean Code*: chapters 1 and 2

**Chapter 1, Clean Code.** Key ideas:
- **There will be code.** No tool or AI removes the need to state requirements precisely, and precise requirements *are* code.
- **Bad code has a cost:** productivity drops as a codebase grows messy. Jetank is still small and clean, and keeping it so is cheaper than cleaning it later.
- **We read more than we write.** Optimize for the reader.
- **The Boy Scout Rule:** leave the code a little cleaner than you found it. In Jetank, that means: a commit that touches a file may come with one small naming or structure fix in it, as a separate commit.

Write your own one-sentence definition of clean code in the journal. You'll revise it in unit 10.

**Chapter 2, Clean That Code!** Martin cleans a piece of code step by step, with tests running after every step. Read it for the *process*, not the Java: small steps, always green, each step named. That process is exactly what you'll do to `cli.py` in unit 3.

**Optional, chapter 17 (AIs, LLMs, and God Knows What).** Martin on programming by prompt. Read it now or whenever you're curious, and compare it with your own experience of building Jetank with Claude.

### Jetank lens: read `pyproject.toml` line by line

| What you see | Why it's there |
|---|---|
| `hatchling` build backend with a comment | The legacy `setup.py` would build the old `jetbot` package with cmake. The comment stops the next maintainer from "fixing" this |
| `requires-python = ">=3.10,<3.11"` | One target platform (JetPack 6). See concept 4 |
| `[project.optional-dependencies] hardware = [...]` | Hardware libraries (`smbus2`, `pyserial`) are only needed on the robot. The laptop installs only `dev` |
| Comment about OpenCV from apt | A PyPI `opencv-python` would shadow the GStreamer-enabled system build. A *why* comment again |
| `[project.scripts] jetank = "jetank.app.cli:main"` | One entry point, in the `app` layer |
| ruff `TC` + `runtime-evaluated-base-classes` | Type-only imports go under `TYPE_CHECKING`, except where pydantic needs them at runtime |
| `[[tool.mypy.overrides]] ignore_missing_imports` for `cv2`, `smbus2`, `serial` | Untyped third-party hardware libraries are a *boundary* (unit 6). They're isolated in `drivers/` |
| `addopts = "... -m 'not hardware'"` | Hardware tests never run by accident on the laptop |

### Jetank lens: the map

```text
src/jetank/
├── errors.py        # exception hierarchy, imported everywhere
├── domain/          # pure models: TrackSpeeds, DriveCommand, BoundingBox, Detection, Track…
├── interfaces/      # Protocols (ports): MotorDriver, ServoBus, Camera, Detector, InferenceEngine…
├── config/          # pydantic-settings: the only place for bus numbers, addresses, limits
├── drivers/         # adapters: pca9685, tb6612_tracks, scs_bus, gst_camera, ssd1306, ina219
│   └── fake/        # in-memory fakes for every interface
├── control/         # behaviour: drive (watchdog), arm, gripper, teleop, status
└── app/             # composition root (wiring.py), CLI, teleop server, video stream
```

The CLAUDE.md file in the repo root explains the rules. It's the equivalent of pluggy's architecture map, and it doubles as the project's coding standard. Read it fully in this unit.

### Exercises

1. **[you] Environment anatomy.** Explore `.venv/`. Find `site-packages`, the `python` executable and the editable-install file pointing to `src/`. Then `ssh jetank` and compare the environment there. Write in your journal why there are two different environments, and how the robot's Python finds the apt-installed `cv2`.
2. **[you] Run everything.** `uv run pytest -q` (green in seconds), `uv run jetank --help`, `uv run jetank config`. Where does the printed configuration come from? Find the YAML example, the environment-variable prefix and the defaults, and write down the precedence order.
3. **[you] Draw the layer map.** Without tools, on paper: one box per subpackage, an arrow for each "imports from". Then check yourself: `rg -n "^from jetank\.|^    from jetank\." src/jetank/control`. Was there an arrow you didn't expect? (Hint: `control` imports from `config`. Is that allowed by CLAUDE.md's diagram? Is it a problem? Write your opinion down; you'll revisit it in unit 4.)
4. **[you] Smallest set of names.** If you wrote a new app that only drives the tracks, which five names do you need? (Look at `app/wiring.py` and `control/drive.py`.)
5. **[you] Read three commits.** `git log --oneline`, then `git show` on `9b776fa`, `32e9a24` and `d120d93`. For each: what problem did it solve, which layers did it touch, and did it include tests? This is how you read Jetank's history like a foreign codebase.

### Resources
- uv docs: <https://docs.astral.sh/uv/>
- Python Packaging User Guide: <https://packaging.python.org/>
- Hynek Schlawack, *Testing & Packaging* (why src layout): <https://hynek.me/articles/testing-packaging/>

---

## Unit 2 – Quality gates: pre-commit, CI and an enforced architecture

> **Neovim step:** Turn on the Python language extra, so ruff and the type checker show problems *inside* the editor. (Step 2.)

### Goals
- Understand each existing gate (ruff, mypy, pytest + coverage) and *why* it exists.
- Add the three missing ones: **pre-commit**, **CI on GitHub Actions**, and an **import contract** that enforces the layer rules.

> **Core path:** exercises 1, 2 and 5. CI (3) and the import contract (4) can wait until before unit 6, when new packages appear.

### Concepts

**1. Shift-left quality.** Catch defects as early as possible: editor, then commit hook, then CI, then review, then the robot. It's the economics of finding a layout error in DRC rather than in silicon. For Jetank the last stage is expensive in a new way: a bug found on the robot may already have moved hardware.

**2. Formatting vs linting.** A formatter (`ruff format`) is mechanical and ends style debates. A linter (`ruff check`) finds bug-prone patterns and enforces conventions.

**3. Static typing.** Type hints are a contract checked by mypy. `strict = true` plus the pydantic plugin means: no untyped functions, no implicit `Any`, and pydantic models checked at construction.

**4. Coverage is a floor, not a goal.** It tells you which lines ran, not whether they were verified. CLAUDE.md asks for ≥ 90 % in `domain`, `control`, `config`. Read the `--cov-report=term-missing` output for *which* lines are missed.

**5. Architecture that is only written down will erode.** CLAUDE.md says "dependencies point downward only." Today nothing checks it. **import-linter** turns the rule into a test.

### *Clean Code*: chapters 6 and 31

**Chapter 6, Formatting.** Vertical and horizontal formatting, team rules over personal taste, and "Uncle Bob's Formatting Rules". In Python, `ruff format` decides most of this for you. What remains *your* job:
- **Ordering inside a module.** Public class or function first, private helpers below, in the order they're called (the stepdown rule, which returns in chapters 7 and 11). Check `control/drive.py`: the public methods come first and `_emergency_stop` last.
- **Blank lines as paragraph breaks** inside longer functions.
- **Module length.** Note what Martin says about file size in the vertical-formatting section. CLAUDE.md says ~300 lines. Two Jetank files are over that limit today (`app/cli.py`, `config/settings.py`). That's for units 3 and 4.
- **Team rules:** for Jetank the "team" is you, Claude and CLAUDE.md. The rules are written down and enforced by tools, which is the point Martin makes.

**Chapter 31, Small Cycles.** Source control history, continuous integration, branches versus toggles, continuous build. This is the "why" behind this unit's tooling. Target: **one command runs every gate**, and CI runs it on every push.

### Jetank lens: what already exists

```bash
cd ~/Projects/Jetank
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
```

- Ruff selects `E F W I B UP N SIM RUF PT ANN TC`. Look up each family at <https://docs.astral.sh/ruff/rules/>.
- `"tests/**" = ["ANN"]` relaxes annotations in tests only. Library code must be fully annotated.
- `ignore = ["ANN401"]` allows `Any` in annotations. CLAUDE.md asks for a comment wherever `Any` appears. Run `rg -n "\bAny\b" src` and check if they all have one.
- `.github/workflows/` only contains the docs deployment inherited from JetBot. There is no test CI yet.

### Exercises

1. **[you] Provoke every gate.** Write a deliberately bad `src/jetank/domain/bad.py`:
   ```python
   import os, sys
   def clamp(speed=[], limit=None):
       if limit == None: return speed
       return max(-limit, min(limit, speed))
   ```
   Run all four commands. Explain each finding in your journal (`B006`, `E711`, `ANN`, `I001`…). Then delete the file.
2. **[you] pre-commit.** Add `pre-commit` to the `dev` extra and create `.pre-commit-config.yaml`. Use ruff's official hooks, the hygiene hooks, and a **local** hook for mypy, so it runs in your venv with the pydantic plugin:
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.14.0              # then run `pre-commit autoupdate`
       hooks:
         - id: ruff-check
           args: ["--fix"]
         - id: ruff-format
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v6.0.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-toml
     - repo: local
       hooks:
         - id: mypy
           name: mypy
           entry: .venv/bin/mypy src
           language: system
           pass_filenames: false
           types: [python]
   ```
   Explain in your journal why mypy runs as a local hook here, while pluggy used `mirrors-mypy`. (Hint: what does mypy need to import to check pydantic models?)
3. **[you] CI.** Create `.github/workflows/ci.yml` that runs on push and pull request, on `ubuntu-latest`, with Python 3.10: install with `uv pip install -e ".[dev]"`, then the four gate commands. Push a branch with a failing test, open a PR, watch it go red, fix it. Note which tests are *skipped* in CI (cv2, smbus2) and why that is fine.
4. **[you] Enforce the layers.** Add `import-linter` to the dev extra and a contract to `pyproject.toml`. Start from:
   ```toml
   [tool.importlinter]
   root_package = "jetank"

   [[tool.importlinter.contracts]]
   name = "Layers point downward"
   type = "layers"
   layers = [
       "jetank.app",
       "jetank.control | jetank.drivers",
       "jetank.config | jetank.interfaces",
       "jetank.domain",
   ]
   ```
   Run `uv run lint-imports`. Read the docs on the `|` syntax (independent siblings) and decide if `config` and `interfaces` really belong side by side. Then add a second contract of type `forbidden`: `jetank.control` must not import `jetank.drivers`. Is that still needed, given the `|`? What happens if someone later changes `|` to `:`? Add `lint-imports` to pre-commit and CI.
5. **[you] One command.** Running the gates should take one step, not four. Make one: a `Makefile` target, a `just` recipe, or a short `scripts/check.sh`. Then use it for the rest of the course.

### Resources
- Ruff rules: <https://docs.astral.sh/ruff/rules/>
- mypy cheat sheet: <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>
- pydantic mypy plugin: <https://docs.pydantic.dev/latest/integrations/mypy/>
- pre-commit: <https://pre-commit.com/>
- GitHub Actions with uv: <https://docs.astral.sh/uv/guides/integration/github/>
- import-linter: <https://import-linter.readthedocs.io/>

---

## Unit 3 – Names, functions, comments: refactoring `cli.py`

> **Neovim step:** Code navigation: `gd`, `gr`, `K`, `<C-o>`, project-wide search with `<leader>/`. (Step 3.)

### Goals
- Learn the core rules on naming, functions and comments.
- Apply them by refactoring a real module without changing its behaviour.

> **Core path:** read chapters 3, 4, 5, 7 and 8, then exercises 1 and 4. Chapters 9, 10 and 11 and the other exercises can come later.

### *Clean Code*: chapters 3, 4, 5, 7, 8, 10 (optional: 9, 11)

This is the heart of the book, spread over many short chapters. Take a few lines of notes per chapter. It's fine to stretch this reading over several sittings while you already start exercise 1.

**Chapter 3, First Principles.** "Everything small, well named, organized, and ordered." This is the summary of everything that follows: read it first, then the details.

**Chapter 4, Meaningful Names.**
- *Intention-revealing names:* `age_s` beats `d`. Jetank adds a rule of its own: **units in names** (`timeout_s`, `speed_mps`, `angle_deg`). For robotics that's the most important naming rule there is.
- *Avoid disinformation:* a `servo_list` that is a dict lies.
- *Meaningful distinctions:* `speeds` vs `limited` in `DriveController.set_tracks` (good), `data` vs `info` (bad).
- *One word per concept:* if the camera has `close()`, the detector shouldn't have `release()`. Check all Protocols in `interfaces/` for this.
- *Problem-domain names:* `Track`, `Detection`, `watchdog`, `gripper`. A gardener or roboticist should recognize them.

**Chapter 5, Comments.** Martin's view ("compensating for failure"): a comment is often a failure to express yourself in code. Good comments explain *intent*, *warn of consequences*, or *clarify* something you can't change (a hardware register map). Bad ones are redundant, misleading, mandated noise, or commented-out code.

*Python adaptation:* Python's docstrings are not the comments Martin is worried about: they are the *contract* that `help()`, the editor's hover and API docs show to callers. Jetank requires Google-style docstrings on public classes and functions, and that stays. Martin's advice applies to the *inline* comments. See Appendix F.

**Chapter 7, Clean Functions.** *Small!*, the stepdown rule (read the module top-down like a story), and switch statements (bury them behind polymorphism, or at least in one place). "One thing" means one level of abstraction.

**Chapter 8, Function Heuristics.**
- *Few arguments.* Fewer is better; three needs justification. Python helps with **keyword-only arguments** (`def f(*, timeout_s: float)`) and **pydantic models as argument objects**.
- *No flag arguments.* `render(is_test=True)` means the function does two things. Two functions are better.
- **Command Query Separation:** a function either does something or answers something, not both.
- *Prefer exceptions to returning error codes*, and keep `try` blocks small.
- *DRY*, and *side effects* (no hidden ones).
- Note the section "This is too much to constantly keep in mind". Martin agrees: these are heuristics for review, not rules to recite while typing.

**Chapter 10, One Thing.** Extract Method as the main tool, and "what are large functions anyway?". If you can extract a function whose name is not just a restatement of the code, the original did more than one thing.

**Optional: chapter 9 (The Clean Method)** is a worked example of "make it work, then make it right". **Chapter 11 (Be Polite)** is about the reader: the newspaper metaphor, the stepdown rule again, and "the abstraction roller coaster" (jumping between high and low levels inside one function).

### Jetank lens: a good example and a candidate

**Good: `control/drive.py`.** Read it with chapters 7 and 8 in mind.
- The class docstring states the contract: scaling, watchdog, thread-safety, and "does not own the motor driver and never closes it" (an ownership rule, exactly the kind of thing that can't be expressed in code).
- `command()` is one line: it translates and delegates.
- `_emergency_stop` keeps the error handling out of `set_tracks`'s happy path.

**A CQS question.** `set_tracks` *commands* the motors and *returns* the limited speeds. `tick` stops the motors *and* returns whether it did. Both break Command-Query Separation. Both are also convenient for callers and tests. Decide for yourself whether the violation is justified (exercise 3).

**A candidate: `app/cli.py`, 320 lines.** It holds the argument parser, five commands (`config`, `snapshot`, `teleop`, `display`, `gripper`), helper functions that open optional hardware, and image writing. That's several reasons to change in one file, and above the 300-line limit.

### Exercises

1. **[you] Naming audit.** Pick 15 names in `control/` and `domain/`. For each: does the name tell you what it is without reading the body? Are units there? Find two you'd rename and argue why. (Don't rename yet. Unit 8 has the tool for it.)
2. **[you] Comment audit.** In `drivers/scs_protocol.py` and `drivers/pca9685.py`, classify every comment with Martin's categories (informative, intent, warning, redundant…). Hardware drivers need more comments than domain code. Why?
3. **[you] CQS debate.** Write half a page: should `set_tracks` return the limited speeds? Should `tick` return a bool? Sketch the alternative API for one of them. What would the tests look like? Decide, and record the decision (keeping it is a valid answer, if you can say why).
4. **[you] Refactor `cli.py`.** This is the main exercise of the unit, and the core refactoring workflow:
   1. Read `tests/unit/app/test_cli.py`. Is every command's behaviour pinned by a test? Add tests for anything missing *before* touching the code.
   2. Move each command into its own module, e.g. `app/commands/teleop.py`, and keep `cli.py` as parser + dispatch.
   3. After each move, run the full gate. Commit after each green step (small commits).
   4. Finish when `cli.py` is below ~120 lines and every module has one reason to change.
   Then ask Claude for a review using the prompts in Appendix D, *after* you're done.
5. **[you] Read code you didn't write.** In `app/teleop_server.py`, find the longest function. Can you state what it does in one sentence without "and"? If not, sketch (on paper) how you'd split it.

### Resources
- PEP 8: <https://peps.python.org/pep-0008/> · PEP 257 (docstrings): <https://peps.python.org/pep-0257/>
- Google Python style guide, docstrings section: <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>
- Refactoring catalog (Martin Fowler): <https://refactoring.com/catalog/>
- Talk: Raymond Hettinger, *Beyond PEP 8* (PyCon 2015, on YouTube)

---

## Unit 4 – Objects, data and dependency inversion: designing the tracker

> **Neovim step:** Text objects and the dot command. (Step 4.)

### Goals
- Distinguish objects (behaviour, hidden data) from data structures (exposed data, no behaviour), and value objects from entities.
- Understand SRP, OCP and the **Dependency Inversion Principle**, using Jetank's `interfaces/`.
- **Design** the tracker: its types and its interface. No logic yet.

> **Core path:** read chapters 12 and 19 (and Cosmic Python 1), then exercise 3, the tracker design. Splitting `settings.py` (exercise 2) is a good one-sitting job for later.

### Reading
- *Clean Code* chapters 12, 13 and 19.
- Cosmic Python chapters 1–3: <https://www.cosmicpython.com/book/chapter_01_domain_model.html> (domain model), chapter 2 (repository), chapter 3 (coupling and abstractions).

### Concepts

**1. Domain model.** The code that represents the problem (a detected object, a track, a drive command) independent of files, cameras and I2C. It's the longest-lived part of the system.

**2. Value objects vs entities.**
- *Value object:* defined by its values, immutable, compared by value. `BoundingBox`, `Detection`, `TrackSpeeds`. In Jetank: pydantic models with `frozen=True, extra="forbid"`.
- *Entity:* has an identity that persists while attributes change. A tracked cat is the same cat across frames even as its box moves.

**3. Pydantic in the domain.** Cosmic Python suggests dataclasses inside the domain and pydantic at the edges. Jetank deliberately uses pydantic everywhere, because physical limits (`speed in [-1, 1]`, a box inside the frame) are domain rules and pydantic validates them. The cost is speed: creating a pydantic model is slower than a dataclass. At 30 frames per second with a handful of detections, that cost doesn't matter. In the hot path of a decoder with 8,400 candidate boxes, it would (unit 6).

**4. Dependency Inversion.** High-level policy (the guard behaviour) must not depend on low-level details (TensorRT, GStreamer). Both depend on an abstraction owned by the high-level side: `interfaces/detector.py`, `interfaces/camera.py`. In Python these are `typing.Protocol` classes: structural, so a driver doesn't even need to inherit from them.

### *Clean Code*: chapters 12, 13 and 19

**Chapter 12, Objects and Data Structures.**
- *Data abstraction:* expose behaviour, hide representation.
- *Data/object anti-symmetry:* objects make it easy to add new types and hard to add new functions; data structures with functions do the opposite. Neither is always better.
- *Law of Demeter:* a method talks to its friends, not to strangers. `settings.motors.left.pwm_channel` deep inside a controller is a "train wreck". Inject `MotorsConfig` (or just the driver) instead.
- *Data transfer objects:* pure data, no behaviour. Jetank's config sections are DTOs, and that's fine.
- *The OO/procedural trade-off* and *"but what about performance?"*: keep both in mind for unit 6, where numpy arrays (data structures) beat objects in the hot path.

**Chapter 13, Clean Classes.** Note the first section, "Classes and modules versus files". Martin now addresses languages where a module, not a class, is the unit of organisation, which is Python's case. Small classes, measured in responsibilities, not lines. Cohesion: methods use most of the instance variables.

**Chapter 19, The SOLID Principles.**
- *SRP:* one reason to change, or in Martin's sharper wording, one *actor* who asks for changes.
- *OCP:* add a detector by adding a class, not by editing the behaviour.
- *LSP:* every `MotorDriver` (real or fake) must honour the Protocol's contract, including "`close()` is idempotent and leaves the motors stopped". A fake that skips that breaks LSP and makes tests lie.
- *ISP:* small Protocols. Compare `InferenceEngine` (three members) with a hypothetical "Robot" interface that has everything.
- *DIP:* depend on the `Detector` Protocol, never on the TensorRT driver.

### Jetank lens

- **`domain/detection.py`** has `BoundingBox` (value object with real behaviour: `iou`, `area`, `center`), `Detection`, `Track`, `IntruderEvent`, and the `Response` policy enum. Notice that box coordinates are **normalised** (0..1). That makes everything above the detector independent of camera resolution, and so of the IR camera later.
- **`Track` is frozen.** It has a `track_id` (identity) but can't change. So "the tracker updates a track" must mean "the tracker replaces a track with a new one with the same id". Is `Track` a value object or an entity? Both readings are defensible: it's a snapshot (value) of an entity (the cat).
- **`interfaces/inference.py` vs `interfaces/detector.py`.** Two ports at two levels: `InferenceEngine` (tensor in, tensor out, knows nothing about cats) and `Detector` (frame in, detections out). The TensorRT driver implements the first. A `YoloDetector` composes an engine with pre- and post-processing to implement the second. That split is SRP at port level, and it makes the decoding testable without a GPU.
- **`config/settings.py`, 396 lines.** Ten config sections in one module. Do they change for the same reason?
- **The layering question from unit 1.** `control/drive.py` imports `DriveConfig` from `config`. The alternative: control defines what it needs (a small pydantic model or Protocol), and `app` converts config into it. Which is cleaner? Which is *simpler*? Chapter 18, Simple Design (unit 6), has a rule about this tension: YAGNI.

### Exercises

1. **[you] Law of Demeter hunt.** `rg -n "\.\w+\.\w+\.\w+" src/jetank/control src/jetank/app`. Classify each hit: train wreck, or harmless navigation of a data structure (Martin allows the latter)?
2. **[you] Split `settings.py`.** It already has a sibling `servos.py`. Decide on a split by reason-to-change (motors and drive? perception: camera, detector, tracker, events?). Keep `from jetank.config import JetankSettings` working for all callers (a re-export in `__init__.py`, like pluggy's shims). Tests stay green at each step.
3. **[you] Design the tracker (types only).** Write, in `domain/` or `interfaces/`, without implementations:
   - what goes in (detections of one frame + a timestamp), what comes out (current tracks, and *which became confirmed in this frame*), what state persists.
   - a `Tracker` class signature, or a Protocol if you expect a second implementation (a Kalman tracker later?). Use Martin's argument rules: what's the minimal API?
   - use `TrackerConfig` (it exists) for the thresholds.
   Write it as docstrings and signatures with `...` bodies. Then answer: does the tracker belong in `domain/` (pure logic, no time source of its own) or in `control/`? Record the decision as `docs/adr/0001-tracker-placement.md` (context, decision, consequences; see <https://adr.github.io/>).
4. **[pair] Review the design.** Ask Claude to review your design *only* against chapters 13 and 19 and the Law of Demeter (Appendix D, prompt 2). Accept or reject each comment with a reason.

### Resources
- PEP 544 – Protocols: <https://peps.python.org/pep-0544/>
- pydantic models: <https://docs.pydantic.dev/latest/concepts/models/>
- SOLID overview (Robert C. Martin): <https://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html>
- Refactoring Guru, design patterns in Python: <https://refactoring.guru/design-patterns/python>

---

## Unit 5 – Testing disciplines and error handling: building the tracker with TDD

> **Neovim step:** Run tests from inside the editor (neotest). (Step 5.)

### Goals
- Implement the tracker with strict test-driven development.
- Learn pytest properly: fixtures, parametrization, fakes, property tests.
- Understand Jetank's error design.

> **Core path:** read chapters 14 and 15, then exercises 1 and 3 (the tracker). Steps 1–5 of the tracker are enough to move on; 6–7 can follow later.

### Reading
- *Clean Code* chapters 14 and 15.
- Cosmic Python chapters 4–5: <https://www.cosmicpython.com/book/chapter_04_service_layer.html>, <https://www.cosmicpython.com/book/chapter_05_high_gear_low_gear.html>

### *Clean Code*: chapter 14 (Testing Disciplines)

- **Discipline 1, TDD:** write no production code before a failing test, write only enough test to fail, write only enough code to pass. The cycle is minutes long.
- **Discipline 2, TCR ("test && commit || revert"):** run the tests after every small change; green commits automatically, red *reverts* the change. Brutal, and an eye-opener for how small steps can be. Try it for one sitting (exercise 3).
- **Discipline 3, small bundles:** many tiny commits. That also suits interrupted study time: you can stop after any green step.
- **Tests enable the "-ilities":** without tests you can't refactor, so code can only rot.

### *Clean Code*: chapter 15 (Clean Tests)

- **Clean tests:** test code is held to the same standard as production code. Dirty tests make code hard to change, which is the opposite of their job.
- **Domain-specific testing language:** build helpers so a test reads like the requirement. `tests/unit/drivers/scs_helpers.py` is one. For the tracker you'll want something like `det("cat", x=0.4, w=0.1)`.
- **F.I.R.S.T.:** Fast, Independent, Repeatable, Self-validating, Timely.
- **One concept per test.** Don't take "one assert per test" literally; one *concept* is the rule.

### Error handling (no separate chapter in the 2nd edition)

The first edition had a chapter on error handling; the 2nd edition keeps "prefer exceptions to returning error codes" in chapter 8 and spreads the rest. The classic rules are still standard practice:
- **Use exceptions, not return codes.**
- **Provide context:** the message says what failed and with which values.
- **Define exception classes by the caller's needs:** how will someone *handle* it?
- **Define the normal flow** (the Special Case pattern): a detector that sees nothing returns an empty list, not `None`.
- **Don't return `None` for errors.** With strict mypy, `X | None` is visible in the type, so it's less dangerous than in Java. Jetank's rule is still: raise a specific exception instead of returning `None` for an error.
- **Write try-except-finally first** when code can fail. For Jetank, the `finally` is usually "leave the robot safe".

### Jetank lens

- **`errors.py`.** `JetankError` at the root, `HardwareError` → `CameraError`, `ServoError`. A caller can catch "any hardware problem" or exactly one kind.
- **`TerminationRequested` derives from `KeyboardInterrupt`, not from `JetankError`.** Read its docstring. That's error design for the caller's needs: it must *not* be swallowed by `except Exception`, so every `with` block unwinds and every driver stops the robot.
- **`DriveController.set_tracks`:** `except Exception: self._emergency_stop(); raise`. Broad catch, but it re-raises. The only goal is the safe state. Inside `_emergency_stop`, a failure is logged, not raised, so the original exception isn't hidden. Compare with chapter 8's advice to keep error handling separate from the main logic.
- **Injected time.** `DriveController` takes a `Clock`. `tests/unit/control/test_drive.py` advances a fake clock instead of sleeping. That makes the watchdog test instant and deterministic. Your tracker needs the same: timestamps passed in, never `time.monotonic()` inside.

### Exercises

1. **[you] Read tests first.** Read `tests/unit/control/test_drive.py` before rereading `drive.py`. Write down every behavioural rule of the drive controller you can derive from the tests alone. Then check against the code.
2. **[you] Manual mutation testing.** Make one deliberate bug at a time in `control/drive.py`, run the tests, revert with `git checkout`:
   - change `age_s <= ...` to `age_s < ...` in `tick`
   - remove `self._moving = False` in `stop`
   - drop the `raise` after `_emergency_stop()`
   Which tests catch each? If one survives, write the missing test. That's a real contribution.
3. **[you] Build the tracker, TDD.** Implement the design from unit 4, one rule per red-green-refactor cycle. A possible order:
   1. no detections → no tracks
   2. one detection → one new, unconfirmed track
   3. same object next frame (IoU ≥ threshold) → same `track_id`, `hits` + 1
   4. `confirm_hits` reached → confirmed, and reported as *newly* confirmed exactly once
   5. not matched for longer than `max_age_s` → dropped
   6. two objects → two tracks, matched greedily by best IoU
   7. different labels never match
   Commit after every green bar. Write the test helpers first. For one sitting, try TCR (chapter 14): a small script that runs `pytest -q tests/unit/domain` and then `git commit -am wip` on green or `git checkout .` on red. Squash the wip commits afterwards.
4. **[you] Property tests.** With `hypothesis`, test `BoundingBox.iou`: symmetric, always in [0, 1], 1 for a box with itself (if non-empty), 0 for disjoint boxes. Then one property of your tracker: "track ids are never reused".
5. **[you] Error paths.** What should the tracker do with a timestamp older than the previous frame? Decide, test it, and if it raises, add the exception to `errors.py` with a message that carries both timestamps.

### Resources
- pytest fixtures: <https://docs.pytest.org/en/stable/how-to/fixtures.html> · parametrize: <https://docs.pytest.org/en/stable/how-to/parametrize.html>
- Hypothesis: <https://hypothesis.readthedocs.io/>
- Gary Bernhardt, *Boundaries* talk (functional core, imperative shell): <https://www.destroyallsoftware.com/talks/boundaries>
- Martin Fowler, *Mocks Aren't Stubs*: <https://martinfowler.com/articles/mocksArentStubs.html>
- Kent Beck, *Test-Driven Development: By Example* (book)

---

## Unit 6 – Boundaries: the YOLO decoder and the TensorRT adapter

> **Neovim step:** The debugger (nvim-dap). (Step 6.)

### Goals
- Wrap third-party code (TensorRT, raw model outputs) so it can't leak into the rest of the system.
- Separate the pure numeric core (decoding, NMS) from the GPU adapter.
- Learn the rules of simple design (YAGNI first).

> **Core path:** read chapter 26, then exercises 1, 2 and 4. The TensorRT adapter (3) is a fair candidate for a **[review]** shortcut if time is short: the learning is in the boundary, not the CUDA buffers.

### *Clean Code*: chapter 26 (Clean Boundaries)

This chapter could have been written for this unit. Martin's main example is a third-party IoT framework, "lots o' boundaries", which is close to Jetank's situation with TensorRT, GStreamer and the servo bus.
- **Third-party code:** a library's interface is broad and general. Your system needs a narrow, specific one. Wrap the library so that only one module knows it.
- **Define the interface you wish you had,** and write the adapter behind it. Jetank did this already: `InferenceEngine` and `Detector` exist, the TensorRT driver doesn't.
- **Learning tests** (a technique from the first edition's boundaries chapter; see if chapter 26 still mentions it): while exploring a new library, write the experiments as *tests*. They document what you learned and, when JetPack upgrades the library, they tell you what changed.
- Also look at the UI/application boundary section, and compare it with `app/teleop_server.py`: where does the browser end and the robot begin?

### *Clean Code*: chapter 18 (Simple Design)

Kent Beck's rules of simple design, as Martin presents them: **YAGNI**, **covered by tests**, **maximize expression**, **minimize duplication**. YAGNI is the brake: don't build an interface "for later". A Protocol earns its place when there are (or will be) at least two implementations, a real one and a fake. Check `interfaces/` against that.

### Concepts

**1. Functional core, imperative shell.** Pre-processing (resize, normalise, HWC→CHW) and post-processing (decode boxes, filter by confidence, non-maximum suppression, scale to normalised coordinates) are pure functions over numpy arrays. Only `infer()` touches the GPU. The pure parts can be tested on the laptop with synthetic tensors.

**2. Numeric code hygiene.**
- Type arrays with `numpy.typing.NDArray[np.float32]`.
- Document shapes in docstrings (for YOLO at 640×640: `(1, 84, 8400)`, i.e. 4 box values + 80 class scores for 8,400 candidates. Verify this with your engine in exercise 1, don't trust it).
- Keep dtype conversions explicit.
- Vectorise the hot path. Create pydantic `Detection` objects only for the few boxes that survive NMS, not for every candidate (see unit 4, concept 3).

**3. No colour assumptions.** The IR camera comes later and delivers grey frames. The decoder must not care. Pre-processing may convert grey to three channels for a model trained on RGB, but that's a configurable step, not hidden in the decoder.

### Jetank lens

- **`interfaces/inference.py`** defines the boundary: `input_shape`, `infer(batch)`, `close()`. That's all of TensorRT the rest of the code will ever see.
- **`DetectorConfig`** in `config/settings.py` already names the backend (`tensorrt` or `fake`), the engine path, class names, thresholds, and the `responses` policy (cat → chase, person → alert). Its validator rejects responses for labels the model doesn't know. That's a boundary check at config load, not at runtime.
- **`scripts/export_detector.sh`** produces the engine. Read it: what model, what precision, what input size?
- **`drivers/gst_camera.py`** is the model for a boundary driver: it is the only module that knows GStreamer.

### Exercises

1. **[you] Learning tests.** On the robot (`ssh jetank`), with the engine file exported, write `tests/integration/test_tensorrt_learning.py` (marked `hardware`): load the engine, assert input and output shapes and dtypes, time one inference. Keep it as documentation of what you learned. (Inference moves no motors, but CLAUDE.md's rule still applies: Claude runs nothing on the robot without your go-ahead.)
2. **[you] The decoder, TDD.** In a new `perception/` package, write pure functions: `decode_yolo(output, class_names, confidence_threshold) -> ...` and `non_max_suppression(...)`. Test with hand-built tensors: one box, two overlapping boxes of the same class, two of different classes, nothing above threshold. Add `jetank.perception` to the import contract from unit 2.
3. **[pair] The TensorRT engine adapter.** `drivers/tensorrt_engine.py` implementing `InferenceEngine`. This is mostly CUDA buffer boilerplate. A fair split: you write the integration test and the `close()` contract, Claude fills in the buffer handling, you review it line by line.
4. **[you] `YoloDetector`.** Compose `InferenceEngine` + pre-processing + decoder into a class implementing `Detector`. Unit-test it with a `FakeInferenceEngine` that returns a canned tensor.
5. **[you] Debugger.** Set a breakpoint in your decoder and step through a real engine output saved with `np.save` on the robot. Look at the array shapes at each step.
6. **[you] Simple-design audit.** For each Protocol in `interfaces/`, count the implementations (real + fake). Does any Protocol violate YAGNI?

### Resources
- NumPy typing: <https://numpy.org/doc/stable/reference/typing.html>
- TensorRT Python API: <https://docs.nvidia.com/deeplearning/tensorrt/latest/>
- Ultralytics YOLO export (TensorRT): <https://docs.ultralytics.com/integrations/tensorrt/>
- For NMS: read the `torchvision.ops.nms` documentation, then reimplement it in numpy

---

## Unit 7 – Behaviour and concurrency: the guard state machine

> **Neovim step:** Splits and the quickfix list. (Step 7.)

### Goals
- Model the guard behaviour as an explicit state machine with bounded, safe transitions.
- Understand Python threads well enough to keep the control loop responsive.
- Apply *Clean Code*'s concurrency rules.

> **Core path:** read chapter 22, then exercises 1, 3 and 4. The safety tests in exercise 3 are not optional.

### Concepts

**1. Explicit state machines.** The mission says: watch → confirm → approach → retreat to home → cooldown. Write it as states (an `Enum`) and a transition function, not as nested `if`s spread over a loop. The cleanest form here is pure:
```python
def step(state: GuardState, seen: Observation, now_s: float) -> tuple[GuardState, DriveCommand]:
    ...
```
State in, state out, no I/O. Every transition becomes a short test. The loop that calls `step` (the imperative shell) reads the camera and drives the motors. (Returning a tuple is fine *inside* a module; if it crosses a module boundary, CLAUDE.md wants a model, e.g. `GuardDecision`.)

**2. State pattern vs `match`.** Chapters 7 and 12 say to bury `switch` statements behind polymorphism (one class per state). In Python, a `match` on an `Enum` inside one function is often clearer for five states. Try both on paper and choose (exercise 2).

**3. Safety as invariants.** Every approach ends: by time (`max_approach_s`), by distance (estimated from odometry or box size), by losing the target, or by a stop request. After any of these, the next state is *retreat*, never *watch*. People produce *alert*, never *approach*, unless the config says otherwise. A failed camera or detector stops the robot. Write these as tests first.

**4. Visual servoing, briefly.** Steering toward a target: the horizontal offset of the box centre from the image centre gives the turn rate, the box size estimates distance. A proportional controller with clamped output is enough to start. That's pure math, so test it with hypothesis (output always within limits, sign flips with the offset).

**5. Threads in Python.** The GIL lets one thread run Python bytecode at a time, but I/O and most numpy/TensorRT calls release it. So a camera thread plus a control thread works well. Rules:
- The control loop never waits for a frame. It takes the *latest* frame if there is a new one (a one-slot buffer with a lock), otherwise it keeps going.
- Every thread has a stop event and is joined on shutdown.
- Shared state is minimal and behind a lock, or passed as immutable values (frozen pydantic models are perfect for this).

### *Clean Code*: chapter 22 (Concurrency)

Read "Why concurrency?", the "Concurrency Defense Principles", and the "2025 Update and Report from the Field". The defensive principles, applied to Jetank:
- **SRP for concurrency:** keep concurrency code separate from other code. The state machine knows nothing about threads.
- **Limit the scope of shared data**, use copies, keep threads as independent as possible.
- **Know your execution models:** producer–consumer (camera → detector → behaviour) is the one you need.
- **Shutdown is hard:** plan it from the start. For Jetank, shutdown also means "motors stopped, servos safe".
- **Testing threaded code:** make the threaded part as thin as possible, so most logic is tested single-threaded.

### Jetank lens

- **`DriveController`** is thread-safe with one lock, and its docstring says so. Teleop commands and watchdog ticks come from different threads.
- **`app/signals.py`** turns SIGTERM/SIGHUP into `TerminationRequested`, so a dropped SSH session stops the robot.
- **`control/teleop.py`** is an existing behaviour that turns remote commands into drive and arm actions. Read it as a model, and as a contrast: teleop is driven by a human, the guard by observations.

### Exercises

1. **[you] Draw the state machine** on paper first: states, transitions, guards (conditions), and what drive command each state emits. Mark every transition that exists *for safety*. Put the drawing in the journal and in `docs/` as a mermaid `stateDiagram-v2`.
2. **[you] Choose the representation.** Sketch the transition logic once as a `match` on an `Enum`, and once with one class per state. Pick one using chapter 18's rules (maximize expression vs YAGNI). Write the reason in an ADR.
3. **[you] Implement it, TDD**, in `control/guard.py`, with a `GuardConfig` (time and distance bounds, cooldown, which responses trigger approach). The safety tests from concept 3 come first.
4. **[you] The approach controller.** Box → `DriveCommand`. Property tests: output always within limits, turning direction matches the side of the box, no forward motion when the box is larger than the "close enough" threshold.
5. **[pair] The latest-frame buffer.** A small class with `put(frame)` and `get_latest() -> Frame | None`, thread-safe. You write the tests (including one with two real threads and a stop event), then let Claude review it for race conditions.

### Resources
- Python `threading`: <https://docs.python.org/3/library/threading.html>
- David Beazley, *Python Concurrency From the Ground Up* (PyCon 2015 talk, on YouTube)
- `match` statement tutorial (PEP 636): <https://peps.python.org/pep-0636/>
- Refactoring Guru, State pattern: <https://refactoring.guru/design-patterns/state>

---

## Unit 8 – Architecture: wiring `jetank guard` end to end

> **Neovim step:** Rename and code actions (`<leader>cr`, `<leader>ca`). (Step 8.)

### Goals
- Separate constructing the system from using it.
- Build the guard as a walking skeleton in simulation, then on the robot.
- Record design decisions.

> **Core path:** read chapters 25 and 27, then exercises 1–3. Docs (5) can wait for unit 9.

### *Clean Code*: chapters 25 and 27 (optional: 23, 24, 20)

Part III of the book is a condensed version of Martin's *Clean Architecture*.

**Chapter 27, The Clean Architecture.** The **dependency rule**: source code dependencies point inward, toward policy. Map it onto Jetank: `domain` is the centre, `control` holds the policies (drive, guard), `interfaces` are the ports, `drivers` and `app` are the outer ring. Then compare with your import contract from unit 2. Do they say the same thing?

**Chapter 25, Architectural Boundaries.** "What lines do you draw, and when?" and the **plug-in architecture**: the core doesn't know its plug-ins, the plug-ins know the core. In Jetank, every driver is a plug-in to the core, and `app/wiring.py` is where they get plugged in. Construction (wiring) is kept apart from use (running the loop), so the rest of the system can assume everything is already wired.

**Optional:** chapter 23 (the two values of software: behaviour *and* structure, and why keeping options open matters), chapter 24 (independence: use cases, operation, development, deployment), chapter 20 (component cohesion and coupling; relevant if Jetank ever splits into several packages).

Martin's advice, which fits Jetank: use the simplest thing that works. The composition root is plain Python, no DI framework.

### Concepts

**1. Composition root.** One place decides which concrete class backs each interface. In Jetank that's `app/wiring.py`. Swapping real hardware for fakes is a config change (`backend: fake`), not a code change.

**2. Walking skeleton.** Get a thin slice working end to end first: `FakeCamera` → `FakeDetector` (scripted detections) → your tracker → your state machine → `DriveController` → `FakeMotorDriver`. All in `--sim`, all tested. Then replace one fake at a time with the real driver.

**3. Architecture Decision Records.** A short file per significant decision: context, decision, consequences. You already have one or two from units 4 and 7.

### Jetank lens

- **`app/wiring.py`.** Read how fakes are chosen and how resources are opened inside a `contextlib.ExitStack`, so that every opened driver is closed in reverse order, even on error.
- **`app/cli.py` (after your unit-3 refactoring):** each command now lives in its own module. `guard` becomes one more.

### Exercises

1. **[you] Scripted fakes.** A `FakeDetector` that replays a list of detections per frame (a cat entering from the left, staying 2 s, leaving). It's the test fixture for the whole guard.
2. **[you] The guard loop.** A `GuardRunner` (or similar) in `control/` that runs the loop: get latest frame → detect → track → step → drive → tick watchdog. Inject everything, including the clock. Test it with the scripted fake: "the cat appears, is confirmed after N frames, the robot approaches, the approach times out, the robot retreats, cooldown".
3. **[you] Wire it.** `jetank guard` in the app layer, with `--sim`. Run it on the laptop. Then on the robot with the *real camera and detector* but *fake motors* (config change only). Look at the logs: does the state machine react to a real cat (or a printed photo of one)?
4. **[you] Rename safely.** Do the two renames you chose in unit 3 with `<leader>cr`, then confirm with the full gate.
5. **[you] Document the architecture.** `docs/design.md` (one page): the pipeline, the layers, where each new module lives, what runs in which thread. Link the ADRs.

### Resources
- Cosmic Python, chapter 13 (Dependency Injection and bootstrapping): <https://www.cosmicpython.com/book/chapter_13_dependency_injection.html>
- ADRs: <https://adr.github.io/>
- Alistair Cockburn, Hexagonal Architecture: <https://alistair.cockburn.us/hexagonal-architecture/>
- Mermaid state diagrams: <https://mermaid.js.org/syntax/stateDiagram.html>

---

## Unit 9 – Continuous design: harden, field-test, document

> **Neovim step:** Your own config: one keymap and one option you wrote yourself. (Step 9.)

### Goals
- Take the guard from "works in sim" to "works in the garden" in small, safe steps.
- Record intruder events.
- Write the user-facing documentation.

> **Core path:** read chapters 21 and 28, then exercises 1 and 2 (field-test stages 2–4). The garden test (stage 5) only when stages 2–4 are clean.

### *Clean Code*: chapters 21, 16 and 28

**Chapter 21, Continuous Design.** Design isn't a phase before coding; it continues with every change. Read it with your guard code in mind: after units 5–8, which part has started to rot? Refactor it *now*, in small green steps, while it's still cheap. "First make it work, then make it right" (the theme of chapter 9 too), but don't stop at "works".

**Chapter 16, Acceptance Testing.** Tests written in the language of the *requirement*, not the code. For Jetank the requirement is the mission in CLAUDE.md: "a cat appears, the robot approaches for at most N seconds, returns home, cools down; a person is only reported". Your scripted-detector test from unit 8 is an acceptance test. Make it read like that sentence.

**Chapter 28, Harm.** The first chapter of the Craftsmanship part: no harm to society, to function, to structure. For most software, "harm" is figurative. For a robot that drives around a garden with a cat (and maybe children) in it, it isn't. Connect it to the safety stages below.

### Concepts

**1. Hardware test protocol.** CLAUDE.md: hardware tests are marked, run explicitly, **with the tracks lifted off the ground**, and fixtures restore a safe state even on failure. For the guard, go through these stages:
1. sim (laptop)
2. real camera + detector, fake motors (robot, stationary)
3. real motors, tracks lifted
4. on the ground, indoors, short time bounds, you next to the remote stop
5. garden, supervised

Each stage has a checklist and a log in the journal.

**2. Logging vs printing.** Library code uses `logging.getLogger(__name__)`. Only the app configures handlers. State transitions at INFO, every frame at DEBUG.

**3. Documentation layers.** README (what, install, 30-second example), how-to pages, reference from docstrings, changelog for users, ADRs for maintainers.

### Exercises

1. **[you] Event recording.** Implement `EventSink` (the Protocol exists): write an annotated snapshot and a JSON record per confirmed intruder, honouring `EventsConfig.label_cooldown_s`. Unit-test it with `tmp_path` and a fake clock.
2. **[you] Field test, stages 2–4.** Follow the protocol. Every surprise becomes a test (CLAUDE.md: "a bug fix starts with a failing test"). Record each in the journal.
3. **[you] Refine.** Pick the part of the guard that has rotted most and refactor it in tiny steps (chapter 21), committing after each green bar.
4. **[you] Coverage review.** `uv run pytest --cov-report=term-missing`. For each uncovered line in `domain/`, `control/`, `config/`: test it, or say why it's not worth it.
5. **[you] Docs.** Update the README with a "garden guard" section: how to run sim, how to run on the robot, the safety stages. Update CLAUDE.md's hardware table with anything you learned.

### Resources
- Python logging HOWTO: <https://docs.python.org/3/howto/logging.html>
- Diátaxis documentation framework: <https://diataxis.fr/>
- Keep a Changelog: <https://keepachangelog.com/>

---

## Unit 10 – Relentless improvement and your own standards

> **Neovim step:** Git from the editor: lazygit and blame. (Step 10.)

### Goals
- Review the whole codebase with a smell list and Appendix A.
- Turn what you learned into your own written standards.
- Look back.

> **Core path:** exercises 1, 3 and 4. Read chapter 32 and the book's appendix.

### *Clean Code*: chapter 32 and the appendix

**Chapter 32, Relentless Improvement.** Test coverage, mutation testing (you did it by hand in unit 5), semantic stability, cleaning. Improvement is a habit, not a project phase.

**Appendix, The Clean Code Debate.** Martin and John Ousterhout argue about method length, comments and TDD, and each rewrites the other's `PrimeGenerator`. Read it after ten units of your own experience; you'll have opinions now. Note where you side with whom, and why, for `STANDARDS.md`.

**Optional:** chapters 29 (no defect in behaviour or structure) and 30 (repeatable proof), and the rest of Part IV whenever you like.

### A smell list for the review

The 2nd edition has no smells catalogue (the 1st edition's chapter 17). Use this short list, drawn from the chapters you've read, plus Appendix A:
- **Functions:** too many arguments, flag arguments, dead code, a function that does more than one thing or mixes levels of abstraction (chapters 7, 8, 10).
- **Names:** names that hide side effects, missing units, inconsistent words for one concept (chapter 4).
- **Comments:** redundant, outdated, or commented-out code (chapter 5).
- **Duplication:** e.g. similar open/close boilerplate across drivers (chapters 8, 18).
- **Switches** that should be polymorphism, and polymorphism that should have stayed a `match` (chapters 7, 12, unit 7).
- **Magic numbers:** register addresses, PWM values. Do they live in config or as named constants?
- **Tests:** untested boundary conditions, slow tests, tests that test implementation (chapter 15).

### Exercises

1. **[you] Smell review.** Go through every module of `src/jetank` with the smell list above and Appendix A. Make a list of findings, rank them, fix the top five (separate small commits).
2. **[review] Compare with Claude.** *After* your review, run `/code-review` on the codebase (or ask Claude with prompt 4 in Appendix D). Compare the two lists. What did you find that Claude didn't, and the other way round?
3. **[you] `STANDARDS.md`.** One to two pages: your Python standards for robotics work. Tool chain, layout, typing, testing, errors and logging, safety rules, review checklist. Then compare with CLAUDE.md. Where they differ, decide which is right and update the other. (CLAUDE.md is the standard *Claude* follows in this repo. It should say what *you* believe.)
4. **[you] Retrospective.** Reread the journal from unit 1. Which concept had the biggest impact? Which is still fuzzy? Where did you disagree with Martin, and do you still? Revise your unit-1 definition of clean code.
5. **Optional: the pluggy side track.** If you want an open-source contribution, version 2 of this guide (`~/Projects/_old/python-clean-architecture-guide.v2.md`, weeks 6, 7 and 10) still describes the pluggy deep dive and contribution workflow. With ten units of Jetank behind you, it will go faster.

### Where to go next
- Night mode: the IR camera as a second `Camera` driver and config entry, a detector fine-tuned on IR images. A good test of whether the boundaries hold.
- Cosmic Python part II (events, message bus): a natural next architecture if the guard gets notifications, a web dashboard, or several robots.
- John Ousterhout, *A Philosophy of Software Design*: the other side of the book's appendix, in full.

---

## Appendix A – Clean code review checklist (Jetank edition)

**Structure & architecture**
- [ ] Can you state each module's responsibility in one sentence? Is it under ~300 lines?
- [ ] Do dependencies point downward (app → control/drivers → config/interfaces → domain)? Does `lint-imports` pass?
- [ ] Do controllers receive their collaborators through the constructor? Does nothing outside `app/` choose a concrete driver?
- [ ] Are Protocols few, documented, and each implemented at least twice (real + fake)?

**Functions & classes**
- [ ] Does each function do one thing at one level of abstraction?
- [ ] Are non-obvious parameters keyword-only? Are there no flag arguments?
- [ ] Commands vs queries: where a function does both, is it justified?
- [ ] Are value objects frozen (`frozen=True, extra="forbid"`)?

**Naming & docs**
- [ ] Do names reveal intent and carry units (`_s`, `_deg`, `_mps`)?
- [ ] Do public classes and functions have Google-style docstrings with args, returns, raises and units?
- [ ] Do inline comments explain *why* (or a hardware fact), never *what*? Is commented-out code gone?

**Types & data**
- [ ] Does `mypy --strict` pass? Is every `Any` and `type: ignore` justified in a comment?
- [ ] Are physical limits encoded with `Field(ge=, le=)` or validators?
- [ ] Do only pydantic models (no plain dicts or tuples) cross module boundaries?
- [ ] Are hardware IDs, buses and addresses only in config?

**Errors, logging & safety**
- [ ] Are errors specific `JetankError` subclasses with context in the message, and causes kept with `raise … from`?
- [ ] Is there no silent `except`? Is every broad `except` re-raising or documented?
- [ ] Does every resource owner implement `close()` / `__exit__` and leave the robot safe?
- [ ] Is every loop bounded or stoppable? Is every thread stopped and joined on shutdown?
- [ ] Does the control loop never block on I/O (camera, network)?
- [ ] Are signs and directions per track and joint correct and tested?
- [ ] Is time injected (`Clock`), never read directly in controllers?

**Tests**
- [ ] Is behaviour tested through public APIs with fakes, not mocks of internals?
- [ ] Are boundary conditions and error paths tested? Are hardware tests marked and safe on failure?
- [ ] Is the unit suite fast (seconds) and runnable on a laptop without hardware?

**Workflow**
- [ ] Do pre-commit and CI pass?
- [ ] Are commits small, focused, and well described? Are build artefacts out of git?

---

## Appendix B – Glossary

| Term | Meaning |
|---|---|
| **Port / Adapter** | Interface defined by the core (`interfaces/`) / concrete implementation for one technology (`drivers/`) |
| **Protocol** | Structural interface checked by type checkers (PEP 544) |
| **Fake** | Working in-memory implementation of a port, used in tests and `--sim` |
| **Composition root** | The one place that wires concrete classes together (`app/wiring.py`) |
| **DIP** | Dependency Inversion Principle: depend on abstractions owned by the high-level side |
| **SRP / OCP** | Single Responsibility / Open-Closed Principle |
| **CQS** | Command-Query Separation: a function either changes state or returns information |
| **Law of Demeter** | Talk to your direct collaborators, not to their collaborators |
| **Value object** | Immutable object defined by its values (`BoundingBox`) |
| **Entity** | Object with persistent identity (a tracked cat) |
| **Learning test** | Test that documents how a third-party library behaves (*Clean Code*, boundaries) |
| **TCR** | "Test && commit \|\| revert": green commits automatically, red reverts (*Clean Code*, ch. 14) |
| **Boy Scout Rule** | Leave the code a little cleaner than you found it |
| **Watchdog** | Timer that forces a safe state when commands stop arriving |
| **Safe state** | Motors stopped, servos torque-off or holding a known pose |
| **IoU** | Intersection over union: overlap measure between two boxes, 0..1 |
| **NMS** | Non-maximum suppression: drop overlapping boxes of the same object, keep the best |
| **Tracker** | Associates detections across frames into tracks with stable ids |
| **Visual servoing** | Steering a robot from image measurements (here: box position and size) |
| **Walking skeleton** | Thin end-to-end slice through all layers, built first |
| **ADR** | Architecture Decision Record |
| **src layout** | Package placed in `src/` so tests run against the installed package |

---

## Appendix C – Resource library

**Books**
- Robert C. Martin, *Clean Code*, 2nd edition (Addison-Wesley, 2025; your copy): the companion for this version. Read critically (Appendix F). It condenses parts of his *Clean Architecture* (Part III) and *The Clean Coder* (Part IV)
- Percival & Gregory, *Architecture Patterns with Python* (free): <https://www.cosmicpython.com/>. Jetank's architecture is its ports-and-adapters chapter, applied to hardware
- John Ousterhout, *A Philosophy of Software Design*: "deep modules", and the other side of *Clean Code*'s appendix
- Kent Beck, *Test-Driven Development: By Example*
- Luciano Ramalho, *Fluent Python*, 2nd ed.: idiomatic Python in depth
- Brian Okken, *Python Testing with pytest*, 2nd ed.
- Martin Fowler, *Refactoring*, 2nd ed.

**Online**
- Ousterhout and Martin's debate, the online version (the book's appendix is based on it): <https://github.com/johnousterhout/aposd-vs-clean-code>
- Brandon Rhodes, Python Design Patterns: <https://python-patterns.guide/>
- Refactoring Guru: <https://refactoring.guru/>
- Scientific Python Development Guide: <https://learn.scientific-python.org/development/>

**Tool documentation**
- uv <https://docs.astral.sh/uv/> · ruff <https://docs.astral.sh/ruff/> · mypy <https://mypy.readthedocs.io/> · pydantic <https://docs.pydantic.dev/> · pytest <https://docs.pytest.org/> · hypothesis <https://hypothesis.readthedocs.io/> · pre-commit <https://pre-commit.com/> · import-linter <https://import-linter.readthedocs.io/>

**Robot & vision**
- NVIDIA Jetson / JetPack docs: <https://docs.nvidia.com/jetson/>
- TensorRT: <https://docs.nvidia.com/deeplearning/tensorrt/latest/>
- Ultralytics YOLO docs: <https://docs.ultralytics.com/>
- Waveshare JETANK wiki: <https://www.waveshare.com/wiki/JETANK_AI_Kit>

**PEPs worth reading in full**
- PEP 8, PEP 20, PEP 257, PEP 484, PEP 544, PEP 561, PEP 621, PEP 636: all at <https://peps.python.org/>

---

## Appendix D – Working with Claude without outsourcing the learning

Claude Code knows this repo through CLAUDE.md and wrote much of the existing code. That makes it a strong collaborator and a risk for learning: it's always faster to let it write. Use it deliberately. (Martin's chapter 17, "AIs, LLMs, and God Knows What", is a good companion to this appendix.)

"You write, Claude reviews" is the default, not a vow. You have limited time and a robot that should make progress. Shortcuts are fine when they're *decisions*:
- Take a shortcut on plumbing, boilerplate, and anything you've already done once yourself. Keep doing the parts that are the unit's topic yourself (the tracker in unit 5, the boundary in unit 6, the state machine in unit 7).
- When you shortcut, say so ("[review] mode for this one"), and write one line in the journal: what Claude wrote and why you let it.
- Even in [review] mode, read the diff. A shortcut on writing is fine; a shortcut on understanding isn't.

Claude remembers this default across sessions in this repo, so you don't need to repeat it every time; just say when you want a different mode.

### The working agreement

Say which mode you're in when it isn't the default:

| Mode | You | Claude | Use for |
|---|---|---|---|
| **[you]** | design, tests, code | answers questions, gives *hints*, reviews when asked. **No edits** | every learning exercise |
| **[pair]** | design and tests, or the core logic | fills in the agreed part | boilerplate around a learning topic (CUDA buffers, CI YAML) |
| **[review]** | read and judge every line | implements | hardware bring-up, plumbing, things you've already mastered |

A good opening line: *"Learning mode for unit 5: I'm writing the tracker. Don't edit files. When I ask, give me hints, not code, and review against chapters 14 and 15."*

### Review prompts

Ask against a *specific* principle. Generic "is this good?" gets generic answers.

1. *"Review `control/guard.py` against Clean Code chapters 7 and 8 only: function size, levels of abstraction, arguments, side effects, CQS. Quote the line for each finding."*
2. *"Review my tracker design against chapters 13 and 19 and the Law of Demeter. For each finding, say what would change if I followed it, and whether it's worth it in Python."*
3. *"Here are my tests for the tracker. Which behaviours are not tested? Give me the test names only, not the code."*
4. *"Do a smell review of `src/jetank/drivers/` using the smell list in unit 10 of my guide. For each finding: the smell, the location, and a one-line fix."*
5. *"Act as a safety reviewer: find any path in `control/guard.py` where the robot could keep moving after the approach time limit, a stop request, or a detector failure."*
6. *"I disagree with your comment on X because Y. Argue the other side as well as you can."* (Practising disagreement is part of learning.)

### Reading Claude's code

When Claude writes in **[review]** mode, the diff is your reading exercise. Before accepting, you must be able to explain every line. If you can't, ask, or reject the change.

---

## Appendix E – The Neovim track

### Ground rules

- **One step per unit**, about 3–6 keys. Sticky note next to the screen, remove it when you no longer look at it.
- **10 minutes of practice a day** in units 1–4: `:Tutor` (twice), then `:VimBeGood`.
- **Practise on copies.** Your `*_bak.py` files (like `tests/unit/app/test_wiring_bak.py`) are the place for drills. They are never committed, and nobody else deletes them.
- **VS Code is allowed** when the task is mainly thinking. Note in the journal what made you switch.
- **Discover, don't memorize:** `<leader>` and wait (which-key), `<leader>sk` to search keymaps, `:h <topic>`.
- **Start Neovim in the project's environment:** `cd ~/Projects/Jetank && uv run nvim .`, so the language server sees `.venv`.
- **Autoformat:** Jetank is ruff-formatted with line length 100, which pre-commit will enforce from unit 2. You may turn autoformat on for this project, or keep it off and let pre-commit do it.

### Step 1 (unit 1): survive and read

| Keys | Meaning |
|---|---|
| `i` / `Esc` | insert mode / back to normal mode |
| `h j k l`, `w b`, `gg G` | move by char, word, file start/end |
| `:w` `:q` `:wq` `:q!` | save, quit, both, quit without saving |
| `<leader><space>` | find a file by name (fuzzy) |
| `<leader>e` | file tree (neo-tree). `q` closes it |
| `/text` `n` `N` | search in file, next, previous |
| `u` / `<C-r>` | undo / redo |

**Practice:** open `pyproject.toml`, `CLAUDE.md`, `control/drive.py` and `app/wiring.py` only with `<leader><space>`.

### Step 2 (unit 2): the editor as a quality gate

1. `:LazyExtras`, select `lang.python` with `x`, restart. This installs pyright and ruff. Commit the change in `~/.config/nvim`.
2. Check with `:LspInfo` in a Jetank file. Both servers should be attached.

| Keys | Meaning |
|---|---|
| `]d` / `[d` | next / previous diagnostic |
| `<leader>cd` | full diagnostic for the line |
| `<leader>xx` | all diagnostics (Trouble) |

**Practice:** the "provoke every gate" exercise in Neovim. Note: pyright is not mypy, and pyright doesn't run the pydantic mypy plugin. Jetank's gate is mypy, so the two will sometimes disagree. Record one case.

### Step 3 (unit 3): navigate code

| Keys | Meaning |
|---|---|
| `gd` | go to definition |
| `gr` | list references |
| `K` | hover: signature and docstring |
| `<C-o>` / `<C-i>` | jump back / forward |
| `<leader>/` | grep the project |
| `<leader>ss` | symbols in this file |

**Practice:** start in `app/wiring.py`. `gd` on `DriveController`, then on `mix`, then on `TrackSpeeds`, then `<C-o>` back. Put the cursor on `MotorDriver` in `interfaces/motors.py` and press `gr`: who depends on the port? Then do the `cli.py` refactoring with this navigation.

### Step 4 (unit 4): edit with grammar

| Keys | Meaning |
|---|---|
| `ciw`, `ci(`, `ci"` | change word / inside parens / inside quotes |
| `daf`, `yaf` | delete / copy a whole function |
| `.` | repeat the last change |
| `p` / `P` | paste after / before |
| `o` / `O` | new line below / above and insert |

**Practice:** write the tracker design (signatures and docstrings) in Neovim only.

### Step 5 (unit 5): tests in the editor

Enable the `test.core` extra (neotest with pytest).

| Keys | Meaning |
|---|---|
| `<leader>tr` | run the nearest test |
| `<leader>tt` | run all tests in the file |
| `<leader>to` | output of the last test |
| `<leader>ts` | test summary tree |

**Practice:** the tracker's red-green-refactor cycles, with `<leader>tr` after each change. Keep the full gate in a terminal as the source of truth.

### Step 6 (unit 6): the debugger

Enable the `dap.core` extra (nvim-dap-python with debugpy).

| Keys | Meaning |
|---|---|
| `<leader>db` | toggle breakpoint |
| `<leader>dc` | run / continue |
| `<leader>di` / `<leader>dO` / `<leader>do` | step into / over / out |
| `<leader>du` | debugger UI |

**Practice:** step through your YOLO decoder with a saved engine output. If the setup fights you, use `breakpoint()` and `pdb` in a terminal. The trace matters more than the tool.

### Step 7 (unit 7): several files at once

| Keys | Meaning |
|---|---|
| `<leader>\|` / `<leader>-` | split right / below |
| `<C-h> <C-j> <C-k> <C-l>` | move between splits |
| `<leader>bd` | close buffer. `<S-h>` / `<S-l>` previous / next buffer |
| `:cnext` `:cprev` | walk through grep results (quickfix) |

**Practice:** `control/guard.py` left, its tests right, while you build the state machine.

### Step 8 (unit 8): refactor safely

| Keys | Meaning |
|---|---|
| `<leader>cr` | rename a symbol everywhere |
| `<leader>ca` | code actions (ruff fixes, add import) |
| `<leader>cf` | format on demand |

**Practice:** the renames you chose in unit 3, then the full gate.

### Step 9 (unit 9): own your config

One change you wrote yourself in `~/.config/nvim/lua/config/`. For example, a keymap that runs your unit-2 "one command" check in a terminal, or `vim.opt.colorcolumn = "101"` to see ruff's line length. Read `:h lua-guide` first. Commit it.

### Step 10 (unit 10): git inside the editor

| Keys | Meaning |
|---|---|
| `<leader>gg` | lazygit (stage hunks, commit, log). `?` for help |
| `<leader>gb` | blame the current line |
| `]h` / `[h` | next / previous changed hunk |

**Practice:** stage your smell-review fixes hunk by hunk in lazygit, one commit per fix.

### After unit 10

You should be able to build a Jetank feature end to end in Neovim: find the code, write the test, change it, debug, commit. Next topics: macros (`q`), marks, `:s`, and `:h user-manual`.

---

## Appendix F – *Clean Code* in Python: what to adopt, what to adapt

Martin's examples are mostly Java, and his preferences formed there. Python has modules as first-class units, keyword arguments, duck typing, docstrings and a formatter. Here's how the main rules translate.

| Martin's rule | In Jetank / Python |
|---|---|
| **Functions should be very small** (often 4–10 lines) | Adopt the *intent* (one thing, one level of abstraction). Don't chop a clear 25-line function into five 5-line ones that are each used once. Ousterhout calls those "shallow" |
| **0–2 arguments, avoid 3+** | Python's keyword-only arguments make 3–4 arguments readable. Beyond that, group them in a pydantic model (an argument object, as Martin also suggests) |
| **No flag arguments** | Adopt fully |
| **Comments are mostly failures** | Adopt for inline comments. Docstrings are different: they're the caller's contract and are required in Jetank. Hardware drivers need more comments (register maps, datasheet facts, verified measurements) |
| **Prefer polymorphism to switch** | Adapt: for a handful of fixed cases, `match` on an `Enum` is often clearer. Use polymorphism when new cases get added by new code (a new detector, a new camera) |
| **Small classes, many of them** | Martin's chapter 13 now discusses modules versus classes. Python modules can hold functions. A pure function in a module is often better than a class with one method |
| **Don't return null** (1st edition) | With strict mypy, `X \| None` is visible and checked. Jetank still raises specific exceptions for errors, and returns empty collections for "nothing found" |
| **One assert per test** | Adapt to "one concept per test" |
| **Formatting chapter** | Mostly done by `ruff format`. Keep the stepdown order and the file-size discipline |
| **Hungarian notation and prefixes are bad** | Agreed. But **unit suffixes** (`_s`, `_deg`, `_mps`) are not Hungarian notation: they carry meaning the type doesn't |
| **Law of Demeter** | Adopt for objects with behaviour. Navigating plain data (config DTOs, frozen value objects) is fine, as Martin himself says |
| **Plug-in architecture, dependency rule** (ch. 25, 27) | Adopt fully: drivers are plug-ins, `app/wiring.py` plugs them in, and import-linter enforces the rule |
| **TDD always, test first** (ch. 14) | Adopt as the default for domain and control logic. For hardware bring-up, explore first (learning tests), then pin the behaviour with tests |

Where Martin and Ousterhout disagree most (function size, comments, and test-first development), read both sides in the book's appendix, "The Clean Code Debate", and decide for yourself. Record the decision in `STANDARDS.md`.
