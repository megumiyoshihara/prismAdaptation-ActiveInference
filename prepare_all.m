%PREPARE_ALL  Generate every .mat file REPEAT_MAIN needs before it can run.
%
%   Files that already exist are skipped, so re-running is cheap and safe.
%   Set force = true to regenerate them from scratch.
%
%   Step 1 (pretraining) is by far the slowest.
%
%   See also DEFAULT_CONFIG, PREPARE_PRETRAINING, PREPARE_POSITIONS, REPEAT_MAIN.

clear;

cfg   = default_config();
force = false;

fprintf('--- step 1/2: pretrained policies ---\n');
prepare_pretraining(cfg, force);

fprintf('--- step 2/2: initial positions and shortest distances ---\n');
prepare_positions(cfg, force);

fprintf('preparation complete.\n');
