function prepare_pretraining(cfg, force)
%PREPARE_PRETRAINING  Learn pre-trained policy C per target position on the monitor.
%   PREPARE_PRETRAINING(CFG) writes cfg.pretrained_policy_File, holding the
%   "learnedC" stack (Nd x Ns x Nm^2) that the transfer learner mixes.
%
%   Policies are stored in the order loop = (x-1)*Nm + y, so policy k belongs to
%   the target position (floor((k-1)/Nm)+1, mod(k-1,Nm)+1).  REPEAT_MAIN relies
%   on that ordering to find the policy that is optimal under the prism.
%
%   This step is slow: it runs Nm^2 pretraining simulations of 100000 trials
%   each.  It is skipped whenever the output file already exists.
%
%   See also PREPARE_ALL, PREPARE_POSITIONS, PRETRAINING_ATXY_MAIN.

arguments
    cfg   (1,1) struct
    force (1,1) logical = false
end

if ~force && isfile(cfg.pretrained_policy_File)
    fprintf('  skipped %s (already exists)\n', cfg.pretrained_policy_File);
    return
end

Nm = cfg.Nm;
T  = 100000;

A_step   = make_init_A_for_pretrain(Nm, 20);
learnedC = zeros(cfg.Nd, cfg.Ns, Nm^2);

fprintf('  pretraining %d policies, this takes a long time ...\n', Nm^2);
for x = 1:Nm
    for y = 1:Nm
        loop = (x-1)*Nm + y;
        seed = 10*x + y;
        fprintf('    policy %3d/%d: target (%d,%d)\n', loop, Nm^2, x, y);
        preQ = pretraining_atXY_main(Nm, T, 1000, seed, x, y, A_step);
        learnedC(:,:,loop) = preQ.qc;
        close;
    end
end

save(cfg.pretrained_policy_File, "learnedC", "T", '-v7.3');
fprintf('  wrote %s\n', cfg.pretrained_policy_File);
end
