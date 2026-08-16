function reg = data_files(experiment, cfg)
%DATA_FILES  Registry of the data files an experiment reads and writes.
%   REG = DATA_FILES(EXPERIMENT, CFG) returns one entry per file, for
%   EXPERIMENT "main" or "valid".  Each entry has
%     key      the CFG field the name is stored in
%     name     the file name, built as {role}_{experiment}[_{condition}]
%     vars     the variables the file holds
%     io       "read", "write" or "readwrite", as seen by the simulation
%     note     what the file is for
%
%   DEFAULT_CONFIG and VALIDATION_CONFIG call this and copy every NAME into the
%   CFG field named by KEY, so the rest of the code keeps reading
%   cfg.pretrained_policy_File and friends.  The registry itself stays in
%   cfg.files, and
%       struct2table(cfg.files)
%   prints every file of an experiment with what it holds and what it is for.
%
%   Names that depend on values only known once a simulation is running (the
%   per-session snapshots and the video, which carry the target and the seed)
%   are built in MAIN_LEARNER and VALIDATION_LEARNER instead.
%
%   See also DEFAULT_CONFIG, VALIDATION_CONFIG, LOAD_PRETRAINED_POLICY.

arguments
    experiment (1,1) string
    cfg        (1,1) struct
end

switch experiment
    case "main",  reg = main_registry(cfg);
    case "valid", reg = valid_registry(cfg);
    otherwise
        error("data_files:unknownExperiment", ...
              "experiment must be ""main"" or ""valid"", got ""%s"".", experiment);
end
end

%% Main experiment ---------------------------------------------------------
function reg = main_registry(cfg)
% Condition token of the initial positions.
switch cfg.randhand
    case 2,    positions = "hand";
    case 3,    positions = "target";
    case 4,    positions = "MD" + cfg.MD;
    otherwise, positions = "fixed";
end

stem = "result_main_" + cfg.sim_type + "_" + positions + sweep_token(cfg) + part_token(cfg);

reg = entry("pretrained_policy_File", "pretrained_policy_main.mat", ...
    "learnedC, T", "readwrite", ...
    "One pretrained goal prior per target position; PREPARE_PRETRAINING writes it.");

reg(end+1) = entry("initHand_File", "initpos_main_hand.mat", ...
    "Hand", "write", ...
    "Hand positions per session for the fixed target, from PREPARE_POSITIONS.");

reg(end+1) = entry("shortestDisHand_File", "shortestdis_main_hand.mat", ...
    "shortestDis_hand", "readwrite", ...
    "Shortest hand-to-target distance for each session of initHand_File.");

switch cfg.randhand
    case 2
        reg(end+1) = entry("InitPos_File", "initpos_main_hand.mat", ...
            "Hand", "read", ...
            "Hand positions replayed per session; the target is fixed at (7,5).");
        reg(end+1) = entry("shortestDis_File", "shortestdis_main_hand.mat", ...
            "shortestDis_hand", "read", ...
            "Shortest distance matching InitPos_File.");
    case 3
        reg(end+1) = entry("InitPos_File", "initpos_main_target.mat", ...
            "posSet", "readwrite", ...
            "Per-simulation target pair and hand trajectory, shifted along cfg.dir.");
        reg(end+1) = entry("shortestDis_File", "shortestdis_main_target.mat", ...
            "shortestDis_pos", "readwrite", ...
            "Shortest distance matching InitPos_File.");
    case 4
        reg(end+1) = entry("InitPos_File", "initpos_main_MD" + cfg.MD + ".mat", ...
            "posSet", "readwrite", ...
            "Per-simulation target pair at Manhattan distance cfg.MD, plus its shift matrix.");
        reg(end+1) = entry("shortestDis_File", "shortestdis_main_MD" + cfg.MD + ".mat", ...
            "shortestDis_pos", "readwrite", ...
            "Shortest distance matching InitPos_File.");
    otherwise
        % randhand 0 and 1 draw the positions on the fly.
        reg(end+1) = entry("InitPos_File", "", "", "unused", ...
            "randhand 0 and 1 generate the positions in MAIN_LEARNER.");
        reg(end+1) = entry("shortestDis_File", "", "", "unused", ...
            "randhand 0 and 1 generate the positions in MAIN_LEARNER.");
end

reg(end+1) = entry("reference_policy_File", "reference_policy_main.mat", ...
    "optC", "read", ...
    "Reference policy for the (7,5) target; only REWRITE_EXCEL reads it, REPEAT_MAIN calls MAKE_OPTC.");

reg(end+1) = entry("rslt_File", stem + ".mat", ...
    "rslt", "write", "Raw results of the batch, one struct per simulation.");

reg(end+1) = entry("xls_File", stem + ".xlsx", ...
    "absPos, relPos, t, C_Gerror, lambda sheets", "write", ...
    "Summary sheets paper_figures.py reads.");
end

%% Validation experiment ---------------------------------------------------
function reg = valid_registry(~)
% The validation experiment has no blur and no risk-width sweep, so its names
% carry no condition token and there is only ever one batch.
stem = "result_valid";

reg = entry("pretrained_policy_File", "pretrained_policy_valid.mat", ...
    "learnedC, T", "readwrite", ...
    "One pretrained goal prior per cell of the 3x3 field; PREPARE_VALIDATION writes it.");

reg(end+1) = entry("optC_File", "reference_policy_valid.mat", ...
    "optC, X_star", "read", ...
    "Reference policy the learned C is scored against, and its unnormalised form.");

reg(end+1) = entry("rslt_File", stem + ".mat", ...
    "rslt", "readwrite", ...
    "Raw results of the batch; cfg.aggregateOnly re-exports from it.");

reg(end+1) = entry("xls_File", stem + ".xlsx", ...
    "transfer, naive, lambda_Gerror, lambda_opt_Gerror", "write", ...
    "Learning curves paper_figures.py reads.");
end

%% Shared name fragments ---------------------------------------------------
function tok = sweep_token(cfg)
% Conditions appear in the name only when they are not at their default, so an
% ordinary run keeps the short name.
tok = "";
if cfg.blur ~= 0
    tok = tok + "_blur" + cfg.blur;
end
if cfg.risk_divisor ~= 1
    tok = tok + "_riskRangeby" + cfg.risk_divisor;
end
end

function tok = part_token(cfg)
% A batch that covers every simulation is the whole result and needs no suffix.
% A run over a subset of cfg.simNums marks itself with cfg.part, so two partial
% runs cannot overwrite each other.
if isequal(sort(cfg.simNums(:))', 1:cfg.Nsim)
    tok = "";
else
    tok = "_part" + cfg.part;
end
end

function e = entry(key, name, vars, io, note)
e = struct('key', key, 'name', name, 'vars', vars, 'io', io, 'note', note);
end
