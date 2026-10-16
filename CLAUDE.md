# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project context

This repo is a fork of NVIDIA's original **JetBot** (`jetbot/`, notebooks, docker). The goal now is
to drive a **Waveshare JETANK** (tracked chassis + 4-DOF serial-bus-servo arm + camera) from a
**Jetson Orin Nano Developer Kit**.

- **Platform:** JetPack 6.x (L4T R36.4), Ubuntu 22.04, aarch64, **Python 3.10**. Do not use syntax or
  libraries that require a newer Python.
- **Legacy code:** `jetbot/` targets the original Jetson Nano (JetPack 4, `traitlets`, Adafruit
  MotorHAT, global singletons). Treat it as **reference only** — read it to learn register/pin
  details, but do not extend it. New work goes in the `jetank/` package described below.
- **Notebooks** under `notebooks/` are legacy demos. New notebooks may *use* `jetank`, but no logic
  lives in a notebook — anything worth keeping is moved into the package and tested.

### Mission

An autonomous **garden guard**: the JETANK watches the garden, detects intruding animals
(cats above all) and people, and drives toward animals to chase them off.

- **Day first** with the IMX219; an **infrared camera** (plus IR illuminator) is added later for
  night operation. Camera-specific code stays behind the `Camera` interface so the IR camera is a
  new driver and config entry, not a rewrite. Detection must not assume RGB colour.
- **Pipeline:** camera → detector (TensorRT on the GPU) → tracker (confirm across frames) →
  behaviour state machine (watch → confirm → approach → retreat to home → cooldown) → drive.
- **Safety in the garden:** approaches are bounded in time and distance and always end with a
  return toward the home position. There is a remote stop. A missing or failed camera, detector or
  motor stops the robot (fail-safe). People are reported, not chased, unless the config
  explicitly says otherwise.
- **Detector and policy are swappable:** the detector model and the target class list are
  configuration, so a model fine-tuned for IR/night images can be dropped in later.

### Hardware notes (verify before relying on them)

| Subsystem        | Interface                          | Notes |
|------------------|------------------------------------|-------|
| Track motors     | I2C, PCA9685 PWM (max 1526 Hz) + TB6612 H-bridge | **Verified:** PCA9685 at `0x60` on `/dev/i2c-7` (header pins 3/5); `0x70` is its all-call address. Motor channel map/direction not yet verified. |
| Arm / camera pan | Serial bus servos (SCServo/ST)     | UART on header pins 8/10 → `/dev/ttyTHS1` on Orin Nano JP6; user must be in `dialout`. |
| OLED display     | I2C, SSD1306                       | Optional. Responds at `0x3C` on `/dev/i2c-7`. |
| Battery monitor  | I2C, INA219                        | Optional. Responds at `0x41` on `/dev/i2c-7`. |
| Camera           | CSI, IMX219, GStreamer `nvarguscamerasrc` | **Verified** on CAM0 (`/dev/video0`), 1640x1232@30 → 640x480. Wide-angle module shows a pink cast with default ISP tuning. Never block the control loop on frame capture. |

Bus numbers, addresses and device paths are **configuration, never constants in code** (see
`config` below). When something here turns out to be wrong on the real robot, update this table.

## Architecture

New code lives in a `src/`-layout package with strict layering. Dependencies point **downward
only**; nothing below `app` imports from a layer above it.

```
src/jetank/
├── config/        # pydantic-settings models; load from YAML/env; the only place hardware IDs live
├── domain/        # pure pydantic models & value types (TrackCommand, JointAngles, ArmPose, BatteryState…)
│                  # no I/O, no hardware imports — fully testable on any machine
├── interfaces/    # typing.Protocol definitions: MotorDriver, ServoBus, Camera, Display, PowerMonitor
├── drivers/       # concrete hardware implementations of the interfaces (smbus2, pyserial, GStreamer)
│   └── fake/      # in-memory fakes of every interface, used by tests and `--sim` mode
├── control/       # behaviour: drive controller, arm kinematics, safety/watchdog — depends on interfaces only
├── perception/    # camera frames → domain objects (detection, line following, etc.)
└── app/           # composition root: builds config, wires drivers into controllers, CLI entry points
tests/
├── unit/          # domain + control against fakes; no hardware, runs anywhere, fast
├── integration/   # real drivers; marked `@pytest.mark.hardware`, skipped unless on the robot
└── conftest.py
```

Rules:

- **Program to interfaces.** Controllers receive a `MotorDriver`/`ServoBus`/… via their constructor
  (dependency injection). They never instantiate hardware themselves and never import from `drivers/`.
- **One composition root.** Only `app/` knows which concrete driver is used. Swapping real hardware
  for fakes must be a config change, not a code change.
- **No global state.** No module-level singletons, no `SingletonConfigurable`, no hidden I/O at import time.
- **Small modules with one responsibility.** If a file passes ~300 lines or mixes layers, split it.
- **Explicit resource lifecycle.** Anything owning a bus, port or camera implements
  `__enter__`/`__exit__` (or `close()`), and on exit leaves the robot in a **safe state**
  (motors stopped, servos torque-off or holding a known pose).
- **Safety is not optional.** Motor commands are clamped to configured limits in the domain model;
  the drive controller has a watchdog that stops the motors if no command arrives within a timeout.
  Servo targets are validated against per-joint limits before they reach the bus.

## Python style

- **Type hints everywhere**, checked with `mypy --strict` (or `pyright` strict). No untyped `def`s,
  no bare `Any` without a comment explaining why.
- **Pydantic v2 for all structured data**: config, commands, sensor readings, messages crossing a
  layer boundary. Use `Field(..., ge=, le=)` constraints and `field_validator`/`model_validator` to
  encode physical limits (e.g. speed in `[-1.0, 1.0]`, joint angle within calibrated range).
  Prefer `model_config = ConfigDict(frozen=True, extra="forbid")` for value objects.
- Use `pydantic-settings` (`BaseSettings`) for runtime configuration, with an env prefix such as
  `JETANK_`, and a checked-in example YAML under `config/`.
- Plain dicts/tuples do not cross module boundaries — define a model.
- `typing.Protocol` for interfaces; `enum.Enum`/`StrEnum`-style enums or `Literal` for fixed choices.
- Units in names (`speed_mps`, `angle_deg`, `timeout_s`) or in the model's field description.
- `logging` (module-level `logger = logging.getLogger(__name__)`), never `print` in library code.
- Raise specific custom exceptions (`jetank.errors`) instead of returning error codes or `None`.
- Docstrings on public classes/functions (Google style): what, args, returns, raises, units.
- Formatting and linting with **ruff** (`ruff format`, `ruff check`), line length 100.

## Testing

Every change ships with tests. Code without tests is not done.

- **pytest** with `pytest-cov`; target **≥ 90 % line coverage** for `domain/`, `control/`, `config/`.
- Unit tests use the fakes in `drivers/fake/` — they must run on a laptop with no Jetson, no I2C,
  no camera, in well under a few seconds.
- Test behaviour, not implementation: e.g. "watchdog stops motors after timeout", "invalid joint
  angle raises `ValidationError`", "differential-drive mixing of (v, ω) gives expected left/right".
- Use `hypothesis` for property tests on math (kinematics, clamping, mixing) where it adds value.
- Inject time/clocks (e.g. a `Clock` protocol) instead of calling `time.sleep`/`time.monotonic`
  directly in controllers, so timing logic is testable deterministically.
- Hardware tests live in `tests/integration/`, are marked `@pytest.mark.hardware`, and are skipped by
  default. They are run on the robot with `pytest -m hardware` — **with the tracks lifted off the
  ground** — and must always return the robot to a safe state, even on failure (use fixtures with
  teardown).
- A bug fix starts with a failing test that reproduces it.

## Verification commands

Run all of these before declaring a task complete, and report the actual results:

```bash
ruff format --check src tests
ruff check src tests
mypy src
pytest                      # unit tests only (hardware tests skipped)
pytest -m hardware          # only on the robot, only when explicitly asked
```

If a tool isn't installed yet, set it up via `pyproject.toml` (dev extras) rather than skipping it.

## Code review

After implementing a non-trivial change, review it before handing back:

1. Re-read the full diff (`git diff`) as a reviewer, not as the author.
2. Check against this file: layering respected? interfaces used? pydantic models at boundaries?
   limits validated? safe-state on exit/exception? tests cover the new behaviour and edge cases?
3. Look for correctness issues specific to robotics: unit mix-ups, sign/direction errors per
   track/joint, unbounded loops without a stop path, blocking I/O inside control loops, threads
   without shutdown, resources not closed.
4. Use `/code-review` for an independent pass on larger changes, and `/simplify` to remove
   duplication or over-engineering.
5. Summarise to the user what was changed, what was tested, and anything **not** verified
   (especially anything that would only show up on real hardware).

## Workflow

- For new features, propose the module layout and interfaces first, then implement bottom-up:
  domain models → interfaces → fakes → controller + tests → real driver → app wiring.
- Keep commits small and focused; never commit build artefacts (`build/`, `CMakeFiles/`,
  `*.egg-info`, `__pycache__`) — add them to `.gitignore` if missing.
- Never run code that moves motors or servos without the user's explicit go-ahead in that session.
- Ask before changing system configuration on the Jetson (`jetson-io`, udev rules, groups,
  `nvpmodel`, apt packages).
