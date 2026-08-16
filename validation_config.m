function cfg = validation_config(varargin)
%VALIDATION_CONFIG  Default parameters for the 3x3 validation experiment.
%   CFG = VALIDATION_CONFIG() returns the parameter struct every validation
%   entry point (REPEAT_VALIDATION, VALIDATION_LEARNER, PREPARE_VALIDATION)
%   starts from.
%
%   CFG = VALIDATION_CONFIG(NAME, VALUE, ...) overrides individual fields before
%   the derived ones are computed, so
%       cfg = validation_config('targetX', 3, 'targetY', 1)
%   also updates cfg.targetXY.  Overriding a field afterwards leaves the derived
%   fields stale.
%
%   This experiment is deliberately NOT the one MAIN_LEARNER runs:
%     - the grid is 3x3 and the hidden state is the hand position alone
%       (Ns = Nm^2), whereas the main experiment packs hand and target
%       together (Ns = Nm^4);
%     - there is no prism, and the target is fixed;
%     - one simulation trains BOTH learners on the SAME trajectory, so their
%       learning curves are directly comparable.
%
%   See also VALIDATION_LEARNER, REPEAT_VALIDATION, PREPARE_VALIDATION.

%% Model dimensions ------------------------------------------------------
cfg.Nm = 3;              % the field is Nm x Nm
cfg.Nu = 4;              % size of the action u
cfg.Nd = 4;              % dimension of the decision d

%% Simulation batch ------------------------------------------------------
cfg.Nsim     = 100;
cfg.simNums  = 1:100;    % subset actually simulated by this run
cfg.aggregateOnly = 0;   % 1: skip the simulations, re-export cfg.rslt_File only
cfg.simN     = 1;        % index of the single simulation VALIDATION_LEARNER runs
cfg.seedcore = 14030000; % seed = seedcore + 10000000 + simN -> 24030001 ...
cfg.useseed  = 1;

%% Session schedule ------------------------------------------------------
cfg.Nsession = 1000;     % learning curve length
cfg.T        = 10;       % trials per session; every session runs all T of them

%% Task ------------------------------------------------------------------
cfg.targetX = 2;         % fixed target, no prism in this experiment
cfg.targetY = 2;
cfg.handX   = 1;         % start of the hand, used only when randhand == 0
cfg.handY   = 1;
cfg.randhand = 1;        % 0: fixed start, 1: new random start every session
cfg.range   = 1;         % learning partition
cfg.prior   = 0.25;      % prior over the four actions
cfg.risk_range = 0.05;   % risk is 0.5 +/- risk_range

%% Output ----------------------------------------------------------------
cfg.saveMat       = 1;   % 1: save the per-simulation .mat file
cfg.draw          = 0;   % 1: draw the field every trial (very slow)
cfg.record_traces = 0;   % 1: keep the per-trial info_s / info_o trajectories

%% Caller overrides ------------------------------------------------------
% Everything above can be overridden; only the derived fields below are
% recomputed afterwards, so they always agree with the values actually used.
for k = 1:2:numel(varargin)
    cfg.(varargin{k}) = varargin{k+1};
end

%% Derived quantities ----------------------------------------------------
cfg.Ns   = cfg.Nm^2;     % hidden state is the hand position only
cfg.No   = cfg.Ns;
cfg.seed = cfg.seedcore + 10000000 + cfg.simN;

% Index of the target within a policy stack, and the state column mask.
% Derived from targetX/targetY so the two cannot drift apart.
cfg.targetXY    = (cfg.targetX-1)*cfg.Nm + cfg.targetY;
cfg.targetIndex = 1:cfg.Ns;

%% Data files ------------------------------------------------------------
% Names come from the registry, one entry per file.  This experiment has no blur
% and no risk-width sweep, so no condition token ever enters its file names.
% cfg.files keeps the whole table: struct2table(cfg.files) prints it.
cfg.files = data_files("valid", cfg);
for k = 1:numel(cfg.files)
    cfg.(cfg.files(k).key) = cfg.files(k).name;
end
end
