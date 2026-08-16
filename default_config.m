function cfg = default_config(varargin)
%DEFAULT_CONFIG  Default parameters for the prism-adaptation simulation.
%   CFG = DEFAULT_CONFIG() returns the parameter struct every entry point
%   (PREPARE_ALL, REPEAT_MAIN, MAIN_LEARNER) starts from.
%
%   CFG = DEFAULT_CONFIG(NAME, VALUE, ...) overrides individual fields before
%   the input file names are derived from them, so
%       cfg = default_config('MD', 6)
%   picks up the MD6 position files rather than the MD1 ones.  Overriding a
%   field afterwards leaves the derived file names stale.
%
%   This is the only place any parameter of the main experiment is written down.

%% Learner ---------------------------------------------------------------
% "naive"    : the learner builds its goal prior C from scratch.
% "transfer" : the learner mixes pretrained C matrices with weights lambda.
cfg.sim_type = "transfer";

%% Simulation batch ------------------------------------------------------
cfg.Nsim     = 100;      % number of simulated subjects the input files hold
cfg.simNums  = 1:cfg.Nsim; % subset actually simulated by this run
cfg.simN     = 1;        % index of the single simulation MAIN_LEARNER runs
cfg.seedcore = 1220000;
cfg.useseed  = 1;        % 1: seed the RNG with cfg.seed, 0: leave it alone

%% Session schedule ------------------------------------------------------
cfg.Nsession      = 200; % sessions per simulation
cfg.T             = 100; % maximum trials (durations) per session
cfg.exposureStart = 141; % session where the prism is put on
cfg.removalStart  = 171; % session where the prism is taken off

%% Task geometry ---------------------------------------------------------
cfg.Nm    = 10;          % the monitor is Nm x Nm
cfg.Nu    = 4;           % size of the action u
cfg.range = 1;           % learning partition
cfg.dir   = "right";     % prism shift direction: "up", "down", "left", "right"
cfg.dis   = 3;           % prism shift distance, used when the target is fixed
cfg.handX = 2;           % initial hand position, used when randhand == 0
cfg.handY = 8;
cfg.prior = 0.25;        % prior over the four actions

%% Initial hand and target positions -------------------------------------
% 0: hand fixed at (handX, handY), target fixed at (7,5)
% 1: hand drawn at random every session
% 2: hand replayed from the "Hand" array in cfg.InitPos_File
% 3: hand and target replayed from the "posSet" array in cfg.InitPos_File
% 4: same as 3, plus the per-simulation shift matrix posSet(simN).A
cfg.randhand = 4;
cfg.MD       = 1;        % Manhattan distance between baseline and exposure target
% Direction of that shift when randhand == 4: "random" picks any direction at
% distance MD per simulation, "up"/"down"/"left"/"right" fix it.  randhand == 3
% ignores this and uses cfg.dir, because there the shift matrix itself is built
% from cfg.dir (see MAIN_LEARNER and PREPARE_POSITIONS).
cfg.shiftDir = "random";

%% Risk ------------------------------------------------------------------
cfg.risk_range   = 0.05; % base half-width: the risk levels are 0.5 +/- this
cfg.risk_divisor = 1;    % divides that half-width; 1 leaves the risk unchanged

%% Pretrained policy -----------------------------------------------------
% Constant added to every entry of the pretrained stack before the learner sees
% it.  The entries are Dirichlet counts, so this blurs the policy towards
% uniform; 0 leaves it untouched.  See LOAD_PRETRAINED_POLICY.
cfg.blur = 0;

%% Output ----------------------------------------------------------------
cfg.saveMat = 1;         % 1: save the per-simulation .mat file
cfg.record  = 0;         % 1: write a video of the reaching movements (RECORD_VIDEO)
cfg.draw    = 0;         % 1: draw the monitor every trial (implied by record)
cfg.part    = 1;         % marks the output when cfg.simNums covers only part of the batch

%% Caller overrides ------------------------------------------------------
% Applied before the derived fields below, so overriding MD or randhand also
% updates the file names that depend on them.
for k = 1:2:numel(varargin)
    cfg.(varargin{k}) = varargin{k+1};
end

%% Derived quantities ----------------------------------------------------
cfg.Ns = cfg.Nm^4;       % dimension of the hidden state s
cfg.No = cfg.Ns;         % dimension of the sensory input o
cfg.Nd = cfg.Nu;         % dimension of the decision d
cfg.seed = cfg.seedcore + 10000000 + cfg.simN;
if cfg.record == 1
    cfg.draw = 1;        % the video is captured from the figure window
end

% Half-width the learner actually uses.  cfg.risk_range stays the base value so
% that overriding cfg.risk_divisor cannot divide an already divided number.
cfg.risk_halfwidth = cfg.risk_range / cfg.risk_divisor;

%% Data files ------------------------------------------------------------
% Names come from the registry, one entry per file.  cfg.files keeps the whole
% table: struct2table(cfg.files) prints it.
cfg.files = data_files("main", cfg);
for k = 1:numel(cfg.files)
    cfg.(cfg.files(k).key) = cfg.files(k).name;
end
end
