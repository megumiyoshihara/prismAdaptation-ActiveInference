function learnedC = load_pretrained_policy(cfg)
%LOAD_PRETRAINED_POLICY  Pretrained goal priors, blurred by cfg.blur.
%   LEARNEDC = LOAD_PRETRAINED_POLICY(CFG) loads the "learnedC" stack from
%   cfg.pretrained_policy_File and adds cfg.blur to every entry.
%
%   The stack holds Dirichlet counts, so adding a constant pushes the normalised
%   policy towards uniform: cfg.blur = 0 leaves it untouched and larger values
%   blur it further.  Entries of the main stack run from 0.1 to about 31, so the
%   blur bites from a few units upwards.  The validation experiment has no blur
%   setting; without the field the stack is loaded unchanged.
%
%   Every consumer must go through this function.  The simulation acts on the
%   blurred stack, and REPEAT_MAIN and REPEAT_VALIDATION rebuild C from the same
%   stack to measure the generalisation error, so loading it raw in one place
%   and blurred in another would make that error meaningless.
%
%   See also DEFAULT_CONFIG, MAIN_LEARNER, REPEAT_MAIN, PREPARE_PRETRAINING.

arguments
    cfg (1,1) struct
end

loaded   = load(cfg.pretrained_policy_File, "learnedC");
learnedC = loaded.learnedC;

if isfield(cfg, 'blur') && cfg.blur ~= 0
    learnedC = learnedC + cfg.blur;
end
end
