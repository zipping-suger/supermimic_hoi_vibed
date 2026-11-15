# supermimic_hoi

Whole-body human-object interaction for the **Unitree G1 (29 DoF) with two Inspire hands**, trained
with a learned coupled robot/object world model and differentiable actor rollouts. The method is
the BlinkMimic HOI design; the robot, objects and reference motions are
[Weave](https://xiaohu-art.github.io/Weave/)'s.

- **41 actions at 50 Hz.** 29 body joints are residuals on the next reference pose. 12 finger
  drivers are absolute commands in [-1, 1], mapped onto their joint limits and low-pass filtered.
  The 12 Inspire mimic joints follow their driver's target through their own PD drive.
- **Dynamic objects.** Objects move only through contact: no virtual spring, support wrench or
  reference force. Physics is MuJoCo-Warp through mjlab, four 5 ms steps per control step.
- **Coupled world model.** One MLP predicts robot and object increments, conditioned on a one-hot
  object key and the physical context. Imagination uses tensor kinematics and analytic rewards
  only; it never queries contact geometry.
- **Learned contact reward.** A separate small head predicts ten per-finger force scores from
  measured simulator contact; actor updates differentiate through its frozen weights.
- **Actor and critic.** A deterministic actor (motion encoder + history decoder) maximizes a
  32-step imagined return with a bounded bootstrap from a target critic; the critic regresses
  lambda-returns of the same rollouts.
- **One fast path.** Fixed-shape collection and all four learners run as `torch.compile`d
  functions inside CUDA graphs. The learning graphs share one scratch memory pool.

## Layout

| Module | Role |
|---|---|
| `robot.py`, `kinematics.py`, `rotations.py` | G1 + Inspire description, torch FK built from the MJCF, rotation utilities |
| `motion.py` | Reference library: clips, finite-difference velocities, FK of tracked bodies, goals |
| `control.py` | Actions to PD targets (reference skip, finger filter, mimic joints) |
| `mdp.py` | Observation, reward and termination, shared by simulation and imagination |
| `sim.py`, `contacts.py`, `variant_collision.py` | CPU and MuJoCo-Warp physics, physics guard, hand-object contact forces |
| `randomization.py`, `reset_guard.py`, `env.py` | Domain randomization and context, penetration screen for resets, the vector environment |
| `networks.py`, `optim.py`, `learning.py` | World model, contact head, actor, critic; Muon+Adam; returns and the gated update |
| `buffer.py`, `cuda_graph.py`, `runner.py` | Replay, CUDA-graph capture, the training loop |
| `prepare/`, `audit.py`, `evaluate.py`, `render.py`, `train.py` | Command-line tools |

## Setup

Python 3.12 and a CUDA GPU. Versions are pinned in `uv.lock` (torch 2.9.0 cu128, mujoco and
mujoco-warp 3.11.0, mjlab 1.6.0, warp 1.16.0).

```bash
uv sync --all-extras
```

Assets and references are generated from a Weave checkout that has its test reference archives
in `data/test/` (see Weave's README; the archives come from `appolyn/Weave` on Hugging Face):

```bash
uv run python -m supermimic_hoi.prepare --weave /path/to/Weave
```

This writes `data/` (git-ignored, about 70 MB):

- `data/robot/`: `g1_inspire.xml`, compiled from Weave's URDF. Every link is a body, the pelvis
  is free, and collision geometry is the link's convex hull capped at 32 vertices (foot and
  shoulder cylinders become capsules, as in Weave's import). Self-collision is disabled, as in Weave.
- `data/objects/`: one MJCF per object with convex collision parts and an explicit inertial. The
  shapes are the ones Weave simulates. The clothes stand uses Weave's hand-fitted primitive
  colliders (a cylinder base, a capsule pole and eight capsule pegs), converted to convex meshes.
  Every other object uses the convex decomposition that PhysX computes when it loads Weave's USD
  (1 to 32 hulls of at most 32 vertices). Those hulls exist only inside Isaac Sim, so
  `prepare/export_weave_hulls.py` exports them once, with Isaac Sim 5.1, into
  `prepare/weave_hulls.npz`. Mass is Weave's rule, density (200 kg/m^3, large box 25) times the
  collision volume; friction is 1.
- `data/weave600/`: 600 clips (1.37 h at 50 Hz) sampled from Weave's test split with seed 0,
  66 or 67 per object over nine objects, covering all 321 base interactions before taking a second
  approach variant of any. Stored per frame: pelvis pose, 53 joint positions, object pose and Weave's
  per-body contact labels. `manifest.json` lists the clips and the source checksums.

Preparation checks that torch FK on the converted model reproduces Weave's recorded body positions
(largest difference 0.002 mm), which validates the joint order and the URDF conversion. Four
candidate clips whose knee reference wraps through pi are rejected as corrupted and listed in the
manifest.

**Rollouts as references.** Weave also publishes the trajectories its policy executed in simulation
for 1,541 of the 1,605 test clips (`rollout/test/` of the same Hugging Face dataset). With those
archives in the checkout's `data/rollout/test/`, `prepare --split rollout/test` samples from them
instead. The kinematic references overlap the object and the ground: in contact frames the hand is
a median 22 mm inside the object, and the feet reach 0.38 m under the floor. The rollouts were
simulated on the same collision shapes and overlap them by a median 1 mm. Their contact labels are
measured contact (above 1 N), so a finger's contact intent is the contact Weave's policy made.

## Use

```bash
uv run pytest                                                   # CPU tests
uv run python -m supermimic_hoi.audit --config configs/weave600.json
uv run python -m supermimic_hoi.train --config configs/weave600.json
uv run python -m supermimic_hoi.evaluate logs/weave600/last.pt --record sub17_smallbox_034_v03
uv run python -m supermimic_hoi.render --rollout results/evaluation_sub17_smallbox_034_v03.npz --output results/rollout.mp4
uv run python -m supermimic_hoi.render --clip sub17_smallbox_034_v03 --output results/reference.mp4
```

`TrainConfig` in `config.py` is the single list of options; a JSON config and the command-line
flags of `train` override its defaults. `configs/weave600.json` (600 worlds, batches of 1,024)
fits a 16 GB GPU. `configs/weave600_large.json` (4,200 worlds, batches of 8,192) mirrors the
allocation BlinkMimic HOI used on large GPUs and has not been run from this repository.

Training writes `config.json`, `reset_guard.json`, `metrics.jsonl` and checkpoints to `log_dir`.
`--resume` restores networks, normalizers, optimizers and counters; episodes and replay start
fresh. A checkpoint is bound to the dataset, the reference processing and the physical assets.
Both presets log to [SwanLab](https://swanlab.cn) (`swanlab_mode: online`, project `supermimic_hoi`,
run named after the log directory). Run `uv run swanlab login` once; `--swanlab-mode offline` or
`disabled` turns the upload off.

The first run compiles the collection and learning functions, which takes several minutes;
later runs reuse the compiler cache.

## Design notes

**References.** Robot and object share one world frame and are always reset together. Joint
references are projected onto the joint limits before velocities and FK are derived. This changes
25.7 % of frames: finger references beyond the URDF limits in 24.5 % (mostly the thumb mimic
joints, by up to 0.35 rad) and body joints in 1.5 % (by up to 0.19 rad). Weave's per-body contact labels become ten per-finger intents: a finger is
intended to touch while any of its links is labelled in contact.

**Each world owns one clip.** `num_envs` is a multiple of the clip count, and replay sampling
draws a clip uniformly before drawing one of its transitions, so clips whose episodes end early
keep their weight.

**Reset states.** Episodes start at the clip's first frame (25 %), ten frames before a sustained
contact phase (25 %) or uniformly within the clip (50 %). Weave's references are kinematic and
overlap the ground and the object: the deepest overlap exceeds 10 mm in 60 % of frames and reaches
0.38 m (feet under the floor while crouching). Resetting MuJoCo into deep overlap launches bodies,
so every proposed start is moved to the nearest frame of its clip whose overlaps are within
`reset_overlap_tolerance`. How deep is harmless depends on the pair, so there is one tolerance per
kind: 40 mm between a hand and the object (the references grasp with the fingers a median 22 mm
inside it, even in Weave's own collision shapes, and a finger is pushed out against a 1 Nm drive), 100 mm between the robot and the
ground, and 20 mm for everything else (body against object, object against ground). This admits
90 % of all frames and 82 % of the frames in which the reference touches the object, at least 181
per clip. A single 20 mm tolerance admits only 24 % of the contact frames, so episodes almost
never started with the object in hand. `audit` measures the admitted resets on the training
physics.

**Rewards** (weights in `RewardConfig`, each scaled by the 0.02 s step): exponential tracking of
the pelvis, 13 further bodies and the object (position, orientation, velocity); fingertip positions
in the object frame for hands the reference has in contact; the contact score; and costs on the
action rate (full normalized body command), soft joint limits, foot slip and finger command
saturation. The contact score averages the force scores of each hand's intended fingers (0 below
1 N, saturating at 5 N) and takes the worse of the required hands.

**Termination** (Weave's thresholds): pelvis position error above 0.25 m, pelvis tilt error above
0.8, object position error above 0.3 m or object tilt error above 0.3. Tilt is the difference of
the vertical component of the body's z axis, so a yaw error never ends an episode; it is penalized
by the orientation rewards. Imagination uses a smooth surrogate of the same conditions and never
rolls past the end of a clip. Contact is rewarded but is not a termination condition: neither the
world model nor the critic carries a contact state, and `evaluate` reports contact coverage beside
the completion rate.

**Physics guard.** A world whose speeds leave generous numerical limits is restored to its last
valid state inside the step, reported as an invalid transition and reset; it never reaches replay
or normalizer statistics. Nonfinite states, nonfinite targets and exhausted contact or constraint
capacity stop training. An invalid learner update (nonfinite, or a gradient norm above
`max_gradient_norm`) is skipped entirely on the device and then stops training with a checkpoint.

**Randomization.** Robot friction (0.3 to 1.2, once per world), torso centre of mass (once per
world), encoder offsets (0.01 rad, per episode) and observation noise. Ground and object friction
are 1 in training and evaluation; MuJoCo uses the larger friction of a contact pair, so robot
friction changes a contact only above 1. The world model, contact
head and critic see these parameters and the object's mass properties; the actor does not.

### Differences from BlinkMimic HOI

- Hands: 12 driver commands plus 12 software-coupled mimic joints, instead of 44 independent
  finger joints.
- Contact: every contact between a hand link and the object counts. There is no calibrated
  inner-hand contact region for the Inspire hand.
- Contact intent is per finger (from Weave's per-link labels) rather than per hand.
- The world model predicts no hand-contact logits, and losing an intended contact does not end an
  episode.
- Reset states are screened per kind of collision pair (40 mm hand-object, 100 mm robot-ground,
  20 mm otherwise) instead of one 3 mm tolerance, and there are no reset translation corrections or
  scene support fixtures (Weave's scenes are ground plus object).
- Object tracking uses Weave's training weights (2.0, with scales of 0.2 m and 0.3 rad) and Weave's
  object termination (0.3 m, tilt 0.3).
- `max_episode_steps` is 1,000 (Weave's 20 s) so a clip of up to 639 frames can be completed
  from its first frame.
- Legacy recipes, alternative reward modes, adaptive sampling, settle tails, fine-tuning, replay
  offload and the diagnostic tools were not carried over.

## Validation

Measured on one RTX 4080 (16 GB) with `configs/weave600.json`. These checks show that the pipeline
runs and is numerically sound. They do not show that a policy learns the task: no long training
run has been made yet.

| Check | Result |
|---|---|
| `pytest` | 45 passed: 41 on CPU, 4 on the GPU (Warp against CPU physics, physics guard, training with all six CUDA graphs) |
| Preparation | Torch FK reproduces Weave's recorded body positions within 0.002 mm on all 600 clips |
| `audit --steps 10` | 221,287 admissible reset frames restored within 4.8e-6 of the reference; over ten reference-PD steps none was rejected numerically and 2,698 (1.2 %) terminated, all but 16 on the object's position or tilt. Open-loop PD does not hold a carried object: 2.2 % of contact-phase resets without any hand-object overlap terminate, and 4.5 % of those 30 to 40 mm deep |
| Training, 500 iterations | 0.237 s per iteration (12,600 simulator transitions per second), 8.7 GiB device memory, no rejected transition; compilation adds several minutes on the first run |
| The same setup without compilation or graphs | 1.92 s per iteration |
| Learning signals over those 500 iterations | world-model loss 7.7 to 0.48, mean episode length 32 to 91 steps, imagined survival 0.15 to 0.69 |
| `evaluate` on that checkpoint | 1 of 600 clips completed, mean progress 0.45, contact on 25 % of intended hand-steps: the policy follows the approach and then loses the object, as expected after two minutes of training |
