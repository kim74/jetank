# Plan: the toy tidy-up robot (autonomous indoors)

Written 2026-11-21 and revised the same day: first for the new intermediate goal, then for
"camera only, no new sensors for now". Background reading before the implementation starts.
Companion to [`python-clean-architecture-guide.md`](python-clean-architecture-guide.md).

**Goal:** the JETANK drives around the house on its own and looks for small plastic toy animals on
the floor (Schleich-style cows, lions, sheep, …). When it finds one, it drives up, picks it up with
the gripper, carries it to a fixed drop-off place, puts it down, and goes looking for the next one.
It keeps doing that until it finds no more toys or its time or battery runs out. Along the way it
gradually builds a **map of the house** and keeps it up to date, so later versions can explore and
navigate more efficiently.

**Why this goal now:** it's fun for the kids, it can be done in winter, and almost every part of it
is reused by the garden guard later: detector, tracker, steering toward a target, obstacle
handling, markers, the state machine, and safety.

**Decided so far:**

- **The camera is the only sensor for now.** Walls, obstacles and drop-offs are seen in the image.
  A LiDAR, an IMU and shock sensors may come later; the design leaves room for them (section 4),
  but nothing is built for them yet.
- **The house has one stair.** It is kept out by configuration (a no-go marker, later a no-go zone
  on the map), and runs happen where it's blocked off. Camera-based edge detection is an extra
  layer on top, never the only protection.
- **Toys:** a box with all the animals is available, for training data and for gripper tests. Most
  of them fit the gripper; the elephant probably doesn't.

---

## 1. Where you start

| Already done | Missing |
|---|---|
| Track drive with speed cap and watchdog | A detector that **knows toy animals** (trained on your box of toys, see M1) |
| Camera driver, live video in the browser | The YOLO output decoder and a TensorRT adapter (interfaces exist, nothing implements them yet) |
| Teleop page with stop, sleep and arm jogging | Tracker (only `TrackerConfig` exists) |
| Arm controller with limits, rest and park; gripper with a torque cap and position readback | **Grasping**: arm poses, a grasp sequence, a check that the grasp worked |
| Battery voltage (INA219), servo temperature and position readback | **Seeing walls, obstacles and drop-offs in the camera image** |
| YOLO11n exported to a TensorRT engine | **Knowing where it is**, and therefore **a map** |
| | Behaviour state machine, the `jetank tidy` command, simulation mode |

What the robot **can't** sense with the current hardware: how far it has actually driven or turned
(tracks slip), and whether it has bumped into something. The INA219 isn't in the motor current
path, so even a stalled motor goes unnoticed. That is why speeds stay low, and why the camera's
view of the floor ahead is so important.

---

## 2. What is different from the garden plan

| Topic | Garden guard (long term) | Toy tidy-up (now) |
|---|---|---|
| Targets | Real animals, COCO classes are enough | Small toy figures at floor level, **a custom-trained detector** |
| Final approach | Stop at a distance | Stop at an exact **grasp position** (within a few cm) |
| Arm | Parked while driving | **Grasp, carry, place**; holds a load for a while (servo heat) |
| Home spot | Return and wait | **Drop-off place** visited many times per run |
| Moving around | Mostly stays near home | **Explores** rooms: seeing walls and the stair matters much more |
| People | Reported, not chased | **Kids nearby**: low speed, gentle gripper, an easy stop |

---

## 3. Milestones

Each milestone is safe and useful by itself. Don't start the next one until the current one works
on the robot. M4 (grasping) doesn't depend on M2–M3 and can be developed in parallel.

### M0: groundwork and measurements (no new software features)

- **Toys and gripper:** go through the box and sort the animals into "fits", "fits only one way"
  (e.g. lying sideways) and "doesn't fit" (the elephant). Measure the maximum jaw opening. Only the
  first two groups are targets; the robot reports the others instead of trying.
- **Camera geometry:** check where the camera sits relative to the gripper and whether it moves
  with the arm. That decides how "the toy is in grasp position" looks in the image, and how image
  rows map to distances on the floor (section 4).
- **Teach the poses** with the teleop page and note the joint positions:
  - *search/drive*: the camera looks at the floor ahead, far enough to see walls and toys;
  - *pre-grasp*: jaws open just above the floor;
  - *grasp*: jaws around the toy;
  - *carry*: toy lifted, camera still sees the floor ahead;
  - *place*.

  These become config values, like the joint calibration already is.
- **The stair:** decide how it's kept out during runs (section 4.3).

### M1: see toys and floor (no motion)

- **Data:** with the robot's own camera, in the *search/drive* pose, record the toys on your floors:
  different rooms, lights, distances and angles, toys lying and standing, single and in groups.
  Record "nothing" scenes too (floor, socks, cables, Lego…) so the detector learns what isn't a
  toy animal. A few hundred labelled images is a good start, and the kids can help arrange the
  scenes.
- **Labels, two kinds on the same images:**
  - boxes for **`toy_animal`** (one class first; species like "lion" or "cow" can come later for
    fun);
  - outlines of the **floor**, where the robot may drive (section 4.1).
- **One network for both:** a small YOLO *segmentation* model (e.g. YOLO11n-seg) can find toy boxes
  and the floor area in the same pass. That saves GPU time compared with two separate models.
- **Train** on a PC or in the cloud, export to ONNX, and build the TensorRT engine on the Jetson
  with the existing `scripts/export_detector.sh`. The model path and the class list are already
  configuration. A labelling tool that runs locally keeps the photos of your home private.
- **Builds:** YOLO decoder (boxes and masks), TensorRT adapter, tracker, a camera thread that always
  hands over the newest frame, and boxes plus floor area drawn over the live video on the page.
- **Measure:** frames per second, the delay from capture to detection, and how far away a toy can
  still be reliably detected. The last number sets the search pattern.

### M2: face the toy (turn in place)

The robot turns until the confirmed toy is in the middle of the image. The horizontal offset gives
the turn rate, with a dead band and a cap. **Measure** how fast the robot actually turns for a
given command on carpet and on hard floor. Without an IMU, this calibration is the only way to know
how far it has turned.

### M3: approach to the grasp position

The robot drives toward the toy and stops when the toy sits in a calibrated **grasp zone** of the
image (a small window around a target position and size).

- Slow down as the toy gets close; the final centimetres are what matter.
- Stop conditions: time limit, estimated distance used up, toy lost, not enough free floor ahead
  (section 4), a no-go marker in view.
- Watch for: the toy disappearing below the camera's view at the last moment. If that happens,
  tilt the camera further down or stop earlier and take the last centimetres blind at a known
  speed.

### M4: grasp and check (robot standing still)

Place a toy in the grasp zone by hand and develop the grasp with the robot not driving:

1. Open the jaws, move to *pre-grasp*, then to *grasp*.
2. Close with the torque cap (already in the gripper controller).
3. **Check:** a closed gripper that stops **before** its "closed" position is holding something,
   and the controller can already read that position back. Then move to *carry* and confirm with
   the camera that the toy has left the floor.
4. If the check fails, open, back off a little, re-aim and try again, at most N times. Then mark
   that toy as "can't pick up" for this run.

Start with the fixed, taught poses. A kinematic model of the arm (forward kinematics, later
inverse) can come later, when you want to grasp toys at varying positions without driving the
robot into an exact spot. Go through the whole box: it shows which animals need a different grasp.

### M5: deliver

- The drop-off place carries a **printed ArUco or AprilTag marker**. The robot turns until it sees
  the marker, approaches it (M3's controller with a different target), moves to *place*, opens,
  lifts the arm, and backs off.
- A **flat, low drop-off zone** (a tray or a mat) is much easier than dropping into a box, given
  the arm's reach.
- While carrying, the arm holds torque. Watch servo temperature (it's already in the servo state)
  and set a limit on carrying time.
- The *carry* pose has to leave the camera a view of the floor ahead. Otherwise the robot drives
  blind while carrying.

### M6: tidy-up mode, without a map

The full behaviour, with simple exploration:

```
            ┌───────────────────────── nothing found for a while / time up / battery low ─────────┐
            ▼                                                                                     │
DONE ◀── RETURN ◀─┐                                                                               │
                  │                                                                               │
EXPLORE ──toy confirmed──▶ APPROACH ──in grasp zone──▶ GRASP ──held──▶ DELIVER ──placed──▶ BACK_OFF ──▶ EXPLORE
   ▲                          │ lost / blocked             │ failed N times
   └──────────────────────────┴────────────────────────────┘ (toy marked "skip")

any state ──fault (camera / detector / motor / servo), remote stop, no-go marker close, floor lost──▶ STOPPED
```

- **Exploration without a map:** drive toward the direction with the most free floor; when free
  floor ahead runs short, turn toward the freest direction (section 4.1); now and then turn in
  place to scan, so toys behind the robot are found too.
- Each state is a small object or function: inputs in (tracks, free-space scan, time, markers,
  gripper state), drive and arm commands out, no I/O inside, clock injected. Every transition gets
  its own unit test.
- **Run limits:** total time, number of toys, battery voltage, servo temperature. When a limit is
  reached, the robot goes back to the drop-off and stops.

### M7: knowing where it is, camera only

See section 4.2. The output is a pose (x, y, heading) with an uncertainty, updated continuously:
estimated from the commands, corrected whenever a marker is in view.

### M8: the house map

- The robot builds a **2D occupancy grid**: a top-down picture of the floor where each cell is
  "free", "occupied" or "unknown". The free-space scan from the camera (section 4.1) fills it, at
  the pose from M7. Each run updates the grid instead of starting over.
- **Gentle updates:** each cell keeps a confidence that changes a little with every new
  observation, so a chair that has moved fades out of the map instead of being a wall forever, and
  a single bad reading doesn't spoil it. With camera-only position estimates the map will be
  rough, and that is fine for "which rooms and corners have I already seen".
- The map is saved to a file after each run (it stays on the robot, nothing goes to the cloud) and
  shown on the browser page, with the robot's position, the drop-off, the markers, and where toys
  were found.
- **No-go zones** (the stair) can be drawn on the map. Once the map is reliable they replace the
  stair's no-go marker; until then, keep both.

### M9: using the map

- **Plan a path** to the drop-off instead of searching for its marker. The marker is then only used
  for the final approach.
- **Explore systematically:** go to the border between known and unknown areas ("frontier
  exploration"), then cover each room in lanes.
- **Remember** where toys tend to be (the kids' room!) and look there first.

### Later: the garden guard

When spring comes: the COCO detector (or a model for real animals), the "approach and come back"
behaviour, the same tracker, steering, safety and markers. The house map doesn't carry over, but
the position and mapping code does.

---

## 4. Seeing the world with the camera only

### 4.1 Walls and obstacles: the free-space scan

The idea: if the camera's height and tilt are known and the floor is flat, **each image row
corresponds to a distance on the floor**. Rows near the bottom are close, rows near the horizon are
far away. So:

1. The segmentation model from M1 marks which pixels are floor.
2. For each image column, find the first pixel from the bottom that is *not* floor. Its row gives
   the distance to the nearest obstacle in that direction, whether that's a wall, a sofa, a foot or
   a toy box.
3. The result looks like a laser scan over the camera's field of view: "free for 1.2 m straight
   ahead, 0.3 m to the left". That's a **virtual range scan**.

Why this is the recommended approach:

- It's honest about distance (in metres, not just "blocked/free"), so the same controller logic
  works for slowing down, turning away and exploring.
- It's exactly what a **LiDAR** delivers later. Behind an interface (say a `FreeSpaceSensor` or
  `RangeScanner` port), the camera fills it today and a LiDAR can replace or add to it without
  touching the behaviour or the map. This is the one place where planning for future sensors pays
  off now.
- It shares the M1 training data and the network with the toy detector.

Simpler fallback for a first try: a small "blocked / free" image classifier, as in the original
JetBot collision-avoidance demo. Quick to train, but it gives no distances and no directions.

Limits to keep in mind:

- The camera sees only forward, over its field of view: nothing behind or beside the robot. So turn
  in place before driving into an unseen direction, and never reverse further than the robot has
  just come.
- The scan is only correct while the camera is in a known pose. Use it in *search/drive* and
  *carry*, and recompute the row-to-distance table if the arm pose changes.
- Glass doors, mirrors and very dark or shiny floors confuse it. Include them in the training data,
  or keep the robot out of those rooms.
- Without a bump sensor nothing notices a collision. Speeds stay low, and the scan's "free"
  distance must stay well above the braking distance.

### 4.2 Knowing where it is without encoders or IMU

- **Dead reckoning:** estimate the motion from the commanded speeds and the time, using the
  calibration from M2 (turn rate per command) and from M3 (distance per second), per floor type.
  It drifts, mainly in heading.
- **Correction with markers:** whenever a marker is in view, its known position fixes the robot's
  pose. The drop-off marker does that already; a handful of extra printed markers on walls or
  furniture, one or two per room, make the estimate usable for a rough map.
- Later, an **IMU** fixes the heading drift (the largest error), and a **LiDAR** makes the
  position good enough for a proper map. Both slot in behind the same pose estimator.

### 4.3 The stair

Since there's only one, keep it out instead of trying to detect it:

1. **Physically** block it off or close its door during runs. This is the most reliable option and
   the one to rely on while testing.
2. **A no-go marker** at the top of the stair: when the robot sees it, it treats the marker as a
   wall a safe distance before the edge and turns away. It's cheap and it fits the marker code from
   M5.
3. **On the map** (M8): a no-go zone around the stair.
4. **The free-space scan** (4.1) as an extra layer: a stair's edge usually looks like floor
   stopping suddenly. Never rely on this alone with a single camera.

---

## 5. Cross-cutting work

- **Safety with kids around:**
  - low `max_speed` indoors, because the robot can't feel a bump;
  - the gripper's torque cap stays (fingers!);
  - a stop that kids can use too: the page's STOP. A physical button can come with the later
    sensors.
- **Low obstacles:** cables and Lego bricks are hard to see. Include them in the "not floor"
  labels, or tidy the room first.
- **Concurrency:** camera thread (newest frame only), inference loop, control loop at a fixed
  10–20 Hz. Each has a clean shutdown, and the control loop never waits on the camera (stale input
  means target lost or STOPPED).
- **Fail-safe:** every input carries a timestamp. If an input is stale or a component throws an
  error, the robot goes to STOPPED. Since the camera is the only sensor, **a frozen or missing
  camera image must stop the robot at once**. Test it deliberately.
- **Simulation (`--sim`):** recorded videos as a fake camera, saved detections and floor masks as a
  fake detector, fake motors and servos. Later, a small 2D floor simulation for developing the map
  on the laptop.
- **Monitoring page:** the teleop page grows into the tidy-up page: state, live video with boxes
  and the floor area, toys collected so far, battery, servo temperature, the map (from M8), and
  STOP and sleep.
- **Battery:** the INA219 voltage decides when the robot heads back and stops.
- **Logs and data:** every grasp attempt (image before and after, success or failure) is saved.
  That's the training data for improving the detector and the grasp.

---

## 6. Testing ladder (each step before the next)

1. Unit tests with fakes: decoder, tracker, free-space scan from a mask, steering, grasp sequence
   and its check, state machine, pose estimate, map updates.
2. Replay: the pipeline on recorded videos, on the laptop and on the Jetson, without motors.
3. Arm on the robot, robot standing still: grasp toys placed by hand, the whole box once (M4).
4. Tracks lifted: check the directions for a toy on the left or right.
5. On the floor, one cleared room, stair blocked off, low speed: turn, approach, grasp, deliver,
   with one toy.
6. Several toys, then the whole room; kids watching from a distance at first.
7. More rooms only with the stair still blocked and the no-go marker tested; the map's no-go zone
   in addition once M8 works.

---

## 7. How it fits the learning guide

| Guide unit | Milestone work |
|---|---|
| 4–5: tracker design and TDD | M1, tracker |
| 6: boundaries (YOLO, TensorRT) | M1, detector and floor mask; the free-space port (camera today, LiDAR later) |
| 7: behaviour and concurrency | M2–M6: steering, grasp sequence, the tidy-up state machine, the loops |
| 8: wiring the command | `jetank tidy`, simulation, the monitoring page |
| 9: field test | the testing ladder |
| after 10 | M7–M9: pose estimate, map, path planning, as a follow-up project |

The guide names the end-to-end command `jetank guard`. For this goal it becomes `jetank tidy`;
the structure is the same.

---

## 8. Decisions

**Made:**

- [x] **Sensors:** camera only for now; LiDAR, IMU and shock sensors maybe later.
- [x] **The stair:** kept out (blocked off during runs, plus a no-go marker), not detected.
- [x] **Toys:** the family's box of animals for training and tests; most fit, the elephant
      probably doesn't.
- [x] **Mission text:** `CLAUDE.md` now describes the tidy-up as the current goal.

**Still open:**

- [ ] **Free space:** floor segmentation and the virtual range scan (recommended), or the simpler
      blocked/free classifier first?
- [ ] **Labels:** one class `toy_animal` first (recommended), or species from the start?
- [ ] **Training:** where to train the detector (your PC with a GPU, or a cloud notebook), and
      which labelling tool.
- [ ] **Drop-off:** a flat tray or mat with a marker (recommended), or a box?
- [ ] **Extra markers** for the pose estimate (M7): how many, and where.
- [ ] **Allowed area:** which rooms the robot may enter.
- [ ] **Who writes what:** the guide puts the tracker, the decoder and the state machine in "you"
      mode, so Claude reviews rather than implements those, unless decided otherwise.
