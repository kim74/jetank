# CLAUDE.md — JETANK Robot Project

This file guides Claude Code (and humans) working in this repository. Read it fully before
writing code. When a rule here conflicts with a convenience, the rule wins; if a rule is
wrong, change this file in the same commit as the code that motivates the change.

---

## 1. Project Overview

Software for the **Waveshare JETANK** tracked robot running on an **NVIDIA Jetson Orin Nano
Developer Kit**. The robot is brought up incrementally — each phase must be working and
tested before the next one builds on it.

| Phase | Goal | Done when |
|---|---|---|
| 0 | Repo, tooling, CI, simulated hardware fakes | `pytest` green on a laptop with no hardware |
| 1 | **Motion control** — tracks via PCA9685 → TB6612 | Robot drives forward/back/turns from CLI; watchdog stops it |
| 2 | **Camera** — CSI camera capture + streaming | Frames captured at target FPS; snapshot & stream CLI |
| 3 | **Telemetry** — battery (INA219), status display | Battery voltage/current logged; low-battery cut-off |
| 4 | **Arm & gripper** — serial bus servos | Arm moves to named poses; gripper open/close; limits enforced |
| 5 | **Teleoperation & data collection** | Remote drive with live video; labelled frames recorded |
| 6 | **Pattern recognition** — lane/line & object detection | Models evaluated offline on recorded data with metrics |
| 7 | **Self-driving** — perception → planning → motion loop | Autonomous lap on test track; safe stop on any failure |

Track the current phase in `docs/roadmap.md`. Do not start code for a later phase without the
user's agreement.

**Language:** Python for everything unless explicitly specified otherwise (e.g. a CUDA kernel
or a shell bring-up script).

---

## 2. Hardware Reference

Source of truth: `docs/hardware/JETANK_Schematic.pdf`. Facts confirmed on the real robot go in
`docs/hardware/hardware.md` (I²C addresses, servo IDs, PWM channel map, calibration values).
**Never hard-code a hardware constant in code without it also being recorded there.**

Known from the schematic (verify on the device with `i2cdetect -y -r <bus>` before relying on it):

- **PCA9685** 16-channel I²C PWM controller — generates PWM *and* direction lines
  (`PWMA/AIN1/AIN2`, `PWMB/BIN1/BIN2`) for the motor driver.
- **TB6612FNG**-style dual H-bridge (`AIN*/BIN*/PWM*/STBY`) driving the two track motors.
- **INA219** current/voltage monitor on I²C (address `0x41` on the schematic) — battery telemetry.
- **Serial bus servos** for the arm/gripper on a half-duplex UART (`SN74LVC1G125` buffer,
  `TXD/RXD/TXEN → DATA`), powered from a dedicated `SERVO 7V` rail.
- **3S Li-ion pack** with protection (`S-8254AA`, `HY2213`) and `SY8286` buck regulator.
- **CSI camera** on the Jetson carrier (captured through GStreamer `nvarguscamerasrc`).

---

## 3. Architecture

Clean Architecture, adapted to robotics. **Dependencies point inward only**:
`infrastructure → application → domain`. The domain never imports hardware libraries,
OpenCV, serial, smbus, or anything with I/O.

```
src/jetank/
├── domain/                  # Pure logic. No I/O, no hardware, no time.sleep(). 100% unit-testable.
│   ├── models/              # Pydantic models, one module per subsystem
│   │   ├── motion.py        #   TrackCommand, Twist, DriveLimits
│   │   ├── camera.py        #   Frame metadata, CameraConfig
│   │   ├── arm.py           #   JointState, ArmPose, GripperState
│   │   ├── power.py         #   BatteryReading, PowerThresholds
│   │   └── perception.py    #   Detection, LaneEstimate, BoundingBox
│   ├── ports.py             # Protocols the domain needs: MotorDriver, Camera, ServoBus, PowerMonitor, Clock …
│   ├── errors.py            # Domain exception hierarchy (JetankError → HardwareError, SafetyError, …)
│   └── services/            # Pure algorithms: differential-drive kinematics, ramping, arm IK,
│                            #   PID, lane-following policy, safety rules
├── application/             # Use cases / orchestration. Uses ports, never concrete drivers.
│   ├── drive.py             #   DriveController (with watchdog), drive_pattern use cases
│   ├── vision.py            #   capture / stream / record use cases
│   ├── manipulation.py      #   move_to_pose, grab, release
│   └── autopilot.py         #   perception → decision → motion loop
├── infrastructure/          # Concrete adapters implementing ports.
│   ├── hardware/            #   pca9685.py, tb6612_motor_driver.py, bus_servo.py, ina219.py, csi_camera.py
│   ├── simulation/          #   In-memory fakes for every port (used by tests and laptop dev)
│   ├── perception/          #   OpenCV / TensorRT model runners
│   └── transport/           #   Web/stream servers, teleop input
├── config/
│   └── settings.py          # pydantic-settings: JetankSettings (env prefix JETANK_, YAML file)
└── cli/
    └── main.py              # Composition root: build adapters from settings, inject, run
tests/                       # Mirrors src/jetank/ (see §6)
scripts/                     # One-off bring-up / calibration scripts (thin; call into the package)
docs/                        # hardware/, bringup/, adr/, roadmap.md
config/                      # default.yaml, robot-specific overrides (no secrets)
```

### Rules

1. **Every hardware dependency sits behind a Protocol** in `domain/ports.py`. Each port has
   at least two implementations: the real driver (`infrastructure/hardware/`) and a fake
   (`infrastructure/simulation/`). A new port without a fake is incomplete.
2. **Composition happens only in `cli/main.py`** (or another entrypoint). Nothing else
   instantiates concrete adapters. Selecting real vs. fake hardware is a settings switch
   (`JETANK_BACKEND=hardware|sim`).
3. **Inject time.** Anything that waits, times out or rate-limits takes a `Clock` port so tests
   run instantly and deterministically.
4. **Low-level drivers are layered:** register-level chip driver (e.g. `PCA9685` — knows
   registers) → device adapter (e.g. `Tb6612MotorDriver` — knows which channels mean what,
   implements `MotorDriver`). Chip drivers depend on a minimal bus Protocol (`I2CBus`,
   `SerialPort`) so they can be tested against a recorded/fake bus.
5. **Infrastructure exceptions never leak.** Catch `OSError`, `serial.SerialException`, etc. at
   the adapter boundary and re-raise as a domain error (`HardwareError`, chained with `from`).
6. **Adding a feature** = new models (domain) → new/extended port → pure service → use case →
   adapter(s) incl. fake → wiring in CLI → tests at each layer → docs. Prefer adding a new
   module over growing an existing one beyond ~300 lines.
7. Record significant design decisions as short ADRs in `docs/adr/NNNN-title.md`
   (context, decision, consequences).

---

## 4. Python Style (Pydantic-first)

- **Target Python:** the Jetson's system Python (JetPack 6 → 3.10). Do not use syntax newer
  than the device runs; confirm with `python3 --version` on the Jetson and record it in
  `docs/hardware/hardware.md`.
- `from __future__ import annotations` at the top of every module.
- **Pydantic v2 for all data crossing a boundary** — configs, commands, sensor readings,
  detections, DTOs between layers. No raw `dict`s or bare tuples across layers.
  - Use `Field(...)` with physical constraints (`ge`, `le`, `gt`) so invalid commands fail
    at construction, e.g. `speed: float = Field(..., ge=-1.0, le=1.0)`.
  - Value objects (commands, readings, poses) are immutable: `model_config = ConfigDict(frozen=True)`.
  - Configuration via `pydantic-settings` `BaseSettings` with `env_prefix="JETANK_"`;
    YAML defaults in `config/`.
  - Use `field_validator` / `model_validator` for cross-field invariants, with a message that
    says *what* is wrong and *what* is allowed.
- **Full type annotations** everywhere, checked with `mypy --strict` (pydantic mypy plugin on).
  Use built-in generics (`list[int]`), `TypeAlias`, `Final` for constants, `Literal`/`Enum`
  for closed sets (`GripperState.OPEN`), `Protocol` for interfaces.
- **Units in names** for every physical quantity: `speed_mps`, `angle_deg`, `duration_s`,
  `voltage_v`, `current_ma`, `pwm_duty` (0–1). No unitless magic numbers.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use `logging` (module-level `logger = logging.getLogger(__name__)`), never `print` outside the CLI.
- Resource-owning adapters (I²C, serial, camera) implement the context-manager protocol and a
  `close()` that is safe to call twice. Motors must be stopped in `close()`.
- Tooling: `uv` for env/deps, `ruff` (lint + format), `mypy --strict`, `pytest`.
  On the Jetson, create the venv with `--system-site-packages` so the JetPack OpenCV
  (built with GStreamer/CUDA) and TensorRT are visible — do **not** pip-install `opencv-python` there.

---

## 5. Safety (non-negotiable)

A robot that moves must fail safe. Every motion-related change must preserve these:

- **Watchdog:** motors stop if no fresh command arrives within `command_timeout_s` (default 0.5 s).
- **Stop on exit:** every entrypoint stops motors and relaxes/holds servos in a `finally`/context
  manager, including on `KeyboardInterrupt` and exceptions.
- **Limits in the domain:** speed/acceleration ramps and servo joint limits are enforced by domain
  services and validated by Pydantic models — never only in the hardware adapter.
- **Low battery:** below the configured cut-off voltage, motion is refused (`SafetyError`).
- **Autonomy** must have a confidence/health check each cycle; on lost perception it stops.
- Hardware tests that move the robot print what they will do and require explicit confirmation
  (or `--yes`) — the robot may be on a table.

---

## 6. Testing (emphasised)

Tests are part of the feature, not an afterthought. **No feature is done without tests at
every layer it touches.**

```
tests/
├── unit/            # Domain + application, using fakes. Fast (<1 s total per module). Always run.
├── integration/     # Adapters against fake buses / recorded data (e.g. PCA9685 register writes,
│                    #   servo packet encoding, perception on images in tests/data/).
├── hardware/        # Runs on the real robot. Marked @pytest.mark.hardware; skipped by default.
└── data/            # Small fixtures (sample frames, recorded packets). Large datasets stay out of git.
```

- Mirror the source tree: `src/jetank/domain/services/kinematics.py` → `tests/unit/domain/services/test_kinematics.py`.
- **Arrange / Act / Assert** blocks, one behaviour per test, named
  `test_<unit>_<condition>_<expected>` (e.g. `test_drive_controller_command_timeout_stops_motors`).
- Mock ports with the fakes in `infrastructure/simulation/` or `MagicMock(spec=Port)`; never
  patch internals of the unit under test.
- Use `pytest.mark.parametrize` for kinematics/limit tables; consider `hypothesis` for
  invariants (e.g. "output duty is always within [-1, 1]").
- Every bug fix starts with a failing test that reproduces it.
- Chip drivers are verified against the datasheet: tests assert the exact register/bytes written.
- Perception (phase 6+): evaluate on a versioned labelled dataset with metrics (precision/recall,
  lateral error) — a model change must not regress them; record results in the PR.
- Each phase has a manual bring-up checklist in `docs/bringup/phase-N.md` (steps, expected
  result, observed result, date) complementing the automated hardware tests.
- Markers: `hardware`, `camera`, `slow`. Default run: `pytest -m "not hardware and not camera"`.
- Coverage target: ≥ 90 % for `domain/` and `application/`; adapters covered by integration tests.

Commands:

```bash
uv run pytest                              # unit + integration (no hardware)
uv run pytest -m hardware                  # on the robot only
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Before declaring a task done: ruff, mypy and pytest must pass. Report failures honestly.

---

## 7. Commenting & Documentation Philosophy

- **Docstrings explain the contract; comments explain the *why*.** Never comment *what* the
  code obviously does.
- Every module starts with a docstring: one-line purpose + `Layer: domain | application |
  infrastructure | entrypoint`, and for hardware modules the chip/datasheet reference.
- Public classes and functions get **Google-style docstrings** (`Args`, `Returns`, `Raises`),
  stating units and valid ranges. Private helpers only when non-obvious.
- Comment hardware specifics at the point of use: register addresses, bit meanings, timing
  requirements, with datasheet section ("PCA9685 §7.3.5: SLEEP must be set before changing PRESCALE").
- Explain magic numbers or replace them with named `Final` constants.
- Mark known gaps as `TODO(phase-N): …` or `FIXME: …` with enough context to act on.
- Keep docs next to what they describe and update them in the same commit: `docs/hardware/`
  for wiring/calibration, `docs/bringup/` for checklists, `docs/adr/` for decisions,
  `README.md` for setup and usage.

---

## 8. Git Workflow

- Remote: `origin` on GitHub. Default branch: `main` — keep it always green.
- One branch per feature/phase step: `feat/motion-watchdog`, `fix/servo-checksum`, `docs/bringup-phase1`.
- **Conventional Commits:** `feat(motion): add differential-drive kinematics`,
  `test(camera): …`, `fix(arm): …`, `docs: …`, `refactor: …`, `chore: …`.
- Small, focused commits; code + tests + docs for a change go together.
- Never commit: virtual envs, `__pycache__`, large datasets, model weights, recordings, secrets,
  or robot-specific network credentials. Use `.gitignore`; large artefacts go in Git LFS or
  external storage referenced from docs.
- Ask before pushing, force-pushing, or rewriting history.

---

## 9. Working Agreement for Claude

- Work phase by phase; propose the file/layer plan before writing a new subsystem.
- Read `docs/hardware/hardware.md` before touching drivers; ask when a hardware fact is unknown
  rather than guessing — wrong PWM channels or servo IDs can damage the robot.
- Development may happen on a laptop without hardware: everything except `tests/hardware/`
  must run there with the simulation backend.
- Don't add dependencies without stating why; prefer small, well-maintained libraries
  (`smbus2`, `pyserial`, `pydantic`, `pydantic-settings`, `numpy`, `opencv` from JetPack).
