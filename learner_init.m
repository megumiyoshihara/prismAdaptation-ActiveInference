function Q = learner_init(cfg, B, D, E, learnedC)
%LEARNER_INIT  Build the initial posterior Q(1) for the configured learner.
%   Q = LEARNER_INIT(CFG, B, D, E, LEARNEDC) returns the session-1 posterior.
%
%   Both learners start from the same likelihood and transition priors; they
%   differ only in the goal prior C:
%     "naive"    learns a single C from scratch (LEARNEDC is ignored).
%     "transfer" keeps the stack of pretrained C matrices in LEARNEDC fixed and
%                learns only the mixing weights lambda, initialised uniformly.
%
%   See also MAIN_LEARNER, LEARNER_POLICY, LEARNER_UPDATE.

No = cfg.No;
Ns = cfg.Ns;
Nd = cfg.Nd;

qaInit = ones(No, Ns, 'single') * 100^(-1);
qaInit = qaInit + eye(No, Ns) * 100;
qbInit = ones(size(B), 'single') * 100^(-1);
qcInit = ones(Nd, Ns, 'single') * 10^(0);

switch cfg.sim_type
    case "naive"
        Q = naive_init(qaInit, qbInit, qcInit, D, E, cfg.T);
    case "transfer"
        Nlambda    = size(learnedC, 3);
        lambdaInit = ones(1, Nlambda, 'single') / Nlambda;
        Q = transfer_init(qaInit, qbInit, qcInit, learnedC, lambdaInit, D, E, cfg.T);
    otherwise
        error("learner_init:unknownSimType", ...
              "sim_type must be ""naive"" or ""transfer"", got ""%s"".", cfg.sim_type);
end
end
