function prepare_validation(cfg, force)
%PREPARE_VALIDATION  Learn one pre-trained policy C per target position on the 3x3 field.
%   PREPARE_VALIDATION(CFG) writes cfg.pretrained_policy_File, holding the
%   "learnedC" stack (Nd x Ns x Ns) the validation transfer learner mixes over.
%
%   Policies are stored in the order loop = (x-1)*Nm + y, matching
%   PREPARE_PRETRAINING on the main side, so policy k belongs to the target
%   (floor((k-1)/Nm)+1, mod(k-1,Nm)+1).  VALIDATION_CONFIG derives cfg.targetXY
%   the same way.
%
%   PREPARE_VALIDATION(CFG, TRUE) regenerates even if the output exists.
%   This step is very slow: Nm^2 pretraining runs of T = 10^7 trials each.
%
%   See also VALIDATION_CONFIG, PRETRAINING_ATXY_VALIDATION, PREPARE_PRETRAINING.

arguments
    cfg   (1,1) struct
    force (1,1) logical = false
end

if ~force && isfile(cfg.pretrained_policy_File)
    fprintf('  skipped %s (already exists)\n', cfg.pretrained_policy_File);
    return
end

Nm = cfg.Nm;
Ns = cfg.Ns;
T  = 10000000;

% Flat likelihood prior for the pretraining runs.
A_step = eye(Ns, Ns)*100^(0) + ones(Ns, Ns)*10^(-2);

learnedC = zeros(cfg.Nd, Ns, Ns);
fprintf('  pretraining %d policies, this takes a very long time ...\n', Ns);
for x = 1:Nm
    for y = 1:Nm
        loop = (x-1)*Nm + y;
        seed = 10*x + y;
        fprintf('    policy %d/%d: target (%d,%d)\n', loop, Ns, x, y);
        preQ = pretraining_atXY_validation(Nm, T, 1000, seed, x, y, A_step);
        learnedC(:,:,loop) = preQ.qc;
        close;
    end
end

save(cfg.pretrained_policy_File, "learnedC", "T", '-v7.3');
fprintf('  wrote %s\n', cfg.pretrained_policy_File);
end
