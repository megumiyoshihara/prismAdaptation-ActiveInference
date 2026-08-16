# Prism adaptation under active inference

Simulation code for a model of prism adaptation built on active inference / the
free energy principle.  An agent reaches for a target on a grid; partway through
the experiment a prism is put on, which shifts where the target *appears*, and
later taken off.  The question the code is written to answer is how the agent
recovers, and whether reusing goal priors it has already learnt for other
targets ("transfer") recovers faster than relearning from scratch ("naive").

Two learners are compared throughout:

| learner | goal prior C | called in the paper |
|---|---|---|
| **transfer** | a mixture of pretrained policies, weighted by `lambda` | proposed |
| **naive** | learnt from scratch as Dirichlet counts `qc` | naive |

The two share the whole simulation loop and differ only in how C is initialised,
evaluated and updated.

---

## Requirements

**MATLAB** — developed and run on R2025b.  The Statistics and Machine Learning
Toolbox is required (`mnrnd`, used for every sampled action and state).  No
other toolbox is used, and there is no `addpath` or `startup.m`: every file sits
flat in the repository root and is called by bare name, so **MATLAB's current
folder must be the repository root**.

**Python 3** — only for the paper figures, not for the simulation:

```
plotly==5.24.1
kaleido==0.2.1
pandas==2.0.3
openpyxl
scipy
```

`kaleido` must stay at 0.2.1.  The 1.x releases expect a real Chrome
installation and cannot write SVG here.

**Platform.**  Everything runs on Linux, Windows and macOS, but two MATLAB
features are not portable and the code works around them:

- `xlswrite` cannot produce Excel format on Linux; it silently falls back to a
  single CSV and every sheet but the first is lost.  All live code therefore
  uses `writematrix`.
- `VideoWriter`'s `'MPEG-4'` profile exists only on Windows and macOS.
  `video_profile.m` selects `'Motion JPEG AVI'` on Linux instead.

---

## Quick start

```matlab
% --- main experiment (10x10 grid, with prism) ---
prepare_all       % generate the input .mat files (once; existing ones are skipped)
repeat_main       % run the batch, write Result_*.mat and Result_*.xlsx

% --- validation experiment (3x3 grid, no prism) ---
prepare_validation(validation_config())
repeat_validation % run, aggregate and write validResult_*.xlsx

% --- video of a single run (optional; run this from the MATLAB desktop) ---
record_video      % writes result/*.avi (Linux) or result/*.mp4 (Windows, macOS)
```

```bash
# --- paper figures, from the .xlsx the MATLAB side wrote ---
python3 paper_figures.py             # every figure -> PA_figure/*.svg
python3 paper_figures.py fig2 fig4   # a subset
python3 paper_figures.py --stats     # the statistical tests, as one table
```

`repeat_main` covers every simulation the input files hold (`cfg.Nsim`, 100 by
default).  Narrow it with `cfg.simNums` to run only part of the batch.

---

## The two experiments

They answer different questions and are deliberately kept separate — they do not
share a config, a learner or an output format.

### Main experiment — `default_config.m`

A 10x10 monitor.  The hidden state packs the hand *and* the target position
together, so `Ns = Nm^4 = 10000`.  200 sessions of up to 100 trials each; the
prism goes on at session 141 and comes off at session 171.  `sim_type` selects
the learner.

Reading order: `default_config` -> `repeat_main` -> `main_learner` ->
`learner_init` / `learner_policy` / `learner_update`.

### Validation experiment — `validation_config.m`

A 3x3 field with no prism and the target fixed at (2,2).  The hidden state is
the hand position alone, `Ns = Nm^2 = 9`.  1000 sessions of exactly 10 trials.

The point of this experiment is to compare the two learners' convergence
directly, so **one simulation updates both of them along the same trajectory**.
The session does not `break` when the target is reached, the action policy stays
uniform, and only `lambda` and `qcValid` carry over between sessions — the rest
of the posterior is reinitialised.  The design isolates the learning of C from
everything else, and lets the session-to-session change be read off a random
walk.

Reading order: `validation_config` -> `repeat_validation` -> `validation_learner`.

---

## Configuring a run

`default_config.m` and `validation_config.m` are the only place parameters are
written down.  Override them by name:

```matlab
cfg = default_config('sim_type', "naive", 'simNums', 1:50);
cfg = default_config('MD', 6);      % also switches to the MD6 input files
```

Overrides are applied **before** the derived fields, so changing `MD`,
`randhand` or `simN` also updates the file names and seeds that depend on them.
Assigning to `cfg.MD` after the call does not — the file names are then stale.

The parameters worth knowing before a first run:

| field | meaning |
|---|---|
| `sim_type` | `"transfer"` or `"naive"` |
| `simNums` | which simulations this run covers (the input files hold `Nsim` = 100) |
| `seedcore` | seed base; each run uses `seedcore + 10000000 + simN`, so a batch is reproducible |
| `Nsession`, `T` | 200 sessions, at most 100 trials each |
| `exposureStart`, `removalStart` | when the prism goes on (141) and comes off (171) |
| `randhand` | how the hand and target are drawn each session — `4` (default) replays the per-simulation positions at Manhattan distance `MD` |
| `MD`, `shiftDir` | distance and direction between the baseline and the exposure target |
| `dir`, `dis` | the prism shift itself, when the target is fixed |
| `blur` | constant added to the pretrained policies, blurring them towards uniform; `0` leaves them alone |
| `risk_divisor` | divides the half-width of the risk levels; `1` leaves the risk alone |
| `saveMat`, `record`, `draw` | write per-session snapshots / a video / draw every trial |

`blur` and `risk_divisor` are the two robustness sweeps behind Figure 5.  The
pretrained policies are Dirichlet counts running from 0.1 to about 31, so adding
a constant washes the policy out towards uniform; the figure sweeps 0 to 50.
The risk levels are `0.5 ± risk_range/risk_divisor`, and the figure sweeps the
divisor from 1 to 100.  Both defaults are exact no-ops, so an ordinary run is
unaffected by their presence.  The validation experiment has neither.

---

## File naming

Every data file the code reads or writes is listed in `data_files.m`.  The
configs copy each name into the field the rest of the code reads
(`cfg.pretrained_policy_File` and friends), and keep the registry itself in
`cfg.files`:

```matlab
cfg = default_config();
struct2table(cfg.files)   % every file of the run, with its variables and purpose
```

The scheme is `{role}_{experiment}[_{condition}]`.  Roles are `pretrained_policy`,
`reference_policy`, `initpos`, `shortestdis`, `result`, `snapshot` and `video`;
experiments are `main` and `valid`.  Conditions appear only when they differ
from their default, so an ordinary run keeps a short name and a swept one says
what was swept:

```
pretrained_policy_main.mat
initpos_main_MD1.mat
shortestdis_main_MD1.mat
result_main_transfer_MD1.xlsx
result_main_transfer_hand_blur30.xlsx
result_main_transfer_hand_riskRangeby20.xlsx
result_valid.xlsx
```

A batch that covers every simulation carries no suffix; a run over a subset of
`cfg.simNums` marks itself `_part{cfg.part}` so two partial runs cannot
overwrite each other.  `paper_figures.py` builds the same names for the
workbooks it reads out of `PA_excel/`.

## Outputs

`repeat_main` writes the raw `rslt` struct array and an `.xlsx` beside it, both
named by the registry.  Every sheet has the same layout: **row 1 is the session
number, column A is the simulation number**, and the block from B2 is the
`Nsim x Nsession` matrix.

| sheet | contents |
|---|---|
| `absPos` | hand x coordinate at the step the shortest path would have finished |
| `relPos` | the same, relative to the target |
| `t` | trials needed to reach the target, per session |
| `C_Gerror` | generalisation error of C against the optimal policy |
| `lambda_Gerror`, `lambda_opt_Gerror` | the same for the mixing weights (transfer only) |
| `truelambdaList`, `shiftedlambdaList` | weight of the exposure and the baseline policy (transfer only) |

`repeat_validation` writes its workbook with `transfer`, `naive`,
`lambda_Gerror` and `lambda_opt_Gerror`.  Its sheets carry one extra column:
**B holds the session-0 baseline and the matrix starts at C2.**

`paper_figures.py` reads these workbooks from `PA_excel/` and writes SVG to
`PA_figure/`.  Selection there is always by label (`.loc`), never by position,
precisely because of the header row and label column.

| figure | contents |
|---|---|
| Fig1 | error decay in a minimal two-state model, against its analytical solution |
| Fig2 | fixed target: reaching error, mixing weights, duration |
| Fig3 | randomised target: the same four panels, plus duration by Manhattan distance |
| Fig4 | validation learning curves, and the main experiment's generalisation error |
| Fig5 | robustness to sensory noise and to the learning rate |

---

## Input data

The `.mat`, `.xlsx` and video files are **not** distributed with the code.  The
simulation inputs are regenerated by the preparation steps:

```matlab
prepare_all                                % main experiment
prepare_validation(validation_config())    % validation experiment
```

Both skip any file that already exists; pass `force = true` (in `prepare_all.m`)
or a second `true` argument to regenerate anyway.

Be aware of what this costs.  The pretraining step dominates: the main
experiment learns one policy per target position, 100 runs of 100000 trials, and
the validation experiment 9 runs of 10^7 trials.  Everything else — the initial
positions and the shortest-distance tables — is quick.

| file | used by | contents |
|---|---|---|
| `pretrained_policy_main.mat` | main, transfer learner | `learnedC [4 10000 100]`, one pretrained policy per target |
| `pretrained_policy_valid.mat` | validation, transfer learner | `learnedC [4 9 9]` |
| `reference_policy_valid.mat` | validation | reference policy `optC [4 9]`, and `X_star` |
| `initpos_main_hand.mat` | main, `randhand 2` | `Hand.s [Nsim Nsession]`, the packed hand-and-target state of each session; the hand shares neither row nor column with the target |
| `initpos_main_target.mat` | main, `randhand 3` | `posSet`, the same fields as the MD file but shifted along `cfg.dir` at `cfg.dis` |
| `initpos_main_MD{MD}.mat` | main, `randhand 4` (default) | `posSet`: per-simulation target pair and hand trajectory |
| `shortestdis_main_hand.mat` | main, `randhand 2` | `shortestDis_hand`, matching `initpos_main_hand.mat` |
| `shortestdis_main_target.mat`, `shortestdis_main_MD{MD}.mat` | main | `shortestDis_pos`, matching the `initpos_` file of the same condition |

Which initial-position file a run reads follows from `cfg.randhand`.  `2` replays
`initpos_main_hand.mat`, whose target is fixed at (7,5) and moves to (4,5) while
the prism is on, so only the hand varies.  `3` and `4` replay a `posSet` — the
baseline and exposure targets of each simulation, the shift matrix `A` between
them, and the hand position per session — from `initpos_main_target.mat` and
`initpos_main_MD{MD}.mat` respectively.  `0` and `1` draw the positions in
`main_learner` and read no file at all.

`prepare_positions` writes the fixed-target pair every time, whatever `randhand`
is set to, so both target regimes the paper compares are prepared in one go.
`make_initHand` fixes the shift inside itself at `"right"` by 3, matching the
defaults of `cfg.dir` and `cfg.dis`.  It takes no direction argument, so running
`randhand 2` with either of those changed leaves the prepared positions
disagreeing with the run: edit the constants in `make_initHand.m` and regenerate
the file.
Each `shortestdis_` file is derived from its `initpos_` file and holds
`[Nsession Nsim]` Manhattan distances; `repeat_main` reads the hand position at
that step to score reaching error.

(`struct2table(cfg.files)` prints the whole list for a given configuration.)

---

## File reference

Everything below sits in the repository root.  `help <name>` in MATLAB prints
the same description as the table.

**Entry points (run these)**

| file | |
|---|---|
| `prepare_all` | generate every input the main experiment needs |
| `repeat_main` | the main batch and its Excel export |
| `prepare_validation` | generate the validation pretrained policies |
| `repeat_validation` | the validation batch, aggregation and export |
| `record_video` | record one transfer run as a video |
| `rewrite_excel` | re-export the sheets from a `Result_*.mat` saved earlier |
| `fig1_simulation` | the minimal two-state model behind Figure 1 |
| `calc_theoreticalGradient` | the analytically predicted convergence rate of the validation experiment |
| `paper_figures.py` | all paper figures and statistics |

**Configuration**

| file | |
|---|---|
| `default_config`, `validation_config` | every parameter of the two experiments |
| `data_files` | the file-name registry both configs resolve their names from |
| `load_pretrained_policy` | loads the pretrained stack and applies `cfg.blur`; every consumer goes through it |

**Simulation**

| file | |
|---|---|
| `main_learner` | one main simulation, either learner |
| `validation_learner` | one validation simulation, both learners at once |
| `learner_init`, `learner_policy`, `learner_update` | the three points where the learners differ |
| `naive_init`, `naive_learning` | naive learner: buffers, and one session of updates |
| `transfer_init`, `transfer_learning` | transfer learner: the same, on the mixture weights |
| `calc_risk` | per-trial risk from the expected free energy |
| `param_normalization` | normalise a Dirichlet parameter matrix |

**Model construction**

| file | |
|---|---|
| `make_Bhand` | transition matrix of the hand on the grid |
| `make_AtargetShift` | likelihood matrix that shifts the perceived target (the prism) |
| `make_optC` | optimal goal prior for one target, the error reference |
| `make_init_A_for_pretrain` | blurred likelihood that seeds the pretraining |
| `calc_xy2S`, `calc_DxS` | pack coordinates into a state, and a state into `kron(delta, s)` |
| `manhattan_distance`, `manhattan_distance_validation` | grid steps from the hand to the target |

**Preparation**

| file | |
|---|---|
| `prepare_pretraining`, `prepare_positions` | the two halves of `prepare_all` |
| `pretraining_atXY_main`, `pretraining_atXY_validation` | learn one policy for one fixed target |
| `make_initHand`, `make_initPos_MD` | initial hand positions for the fixed target (`initpos_main_hand.mat`), and target pairs at distance MD (`initpos_main_MD{MD}.mat`, `initpos_main_target.mat`) |
| `make_randHandS`, `make_randHandS_validation` | a random hand position as a state vector |

**Display**

| file | |
|---|---|
| `figure_output_main`, `figure_output_validation` | hand and target on the grid, plus the policy as colour when one is passed |
| `Cfordisplay`, `Cfordisplay_validation` | lay a policy back onto the grid for drawing |
| `video_profile` | the `VideoWriter` profile usable on this platform |
| `update_disp` | overwrite the progress line in place |

Files whose name ends in `_validation` belong to the 3x3 experiment; their
`_main` counterparts work on the packed hand-and-target state.

---

## Citation

<!-- TODO: paper reference -->

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
