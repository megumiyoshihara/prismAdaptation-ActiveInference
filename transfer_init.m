function Q = transfer_init(qaInit,qbInit,qc1Init, qcPreInit, lambdaInit,D,E,T)
%TRANSFER_INIT  Trial buffers and priors for one session of the transfer learner.
%   Q = TRANSFER_INIT(QAINIT,QBINIT,QC1INIT,QCPREINIT,LAMBDAINIT,D,E,T) allocates
%   the same buffers as NAIVE_INIT, but the policy is a mixture: qc1 is the
%   learner's own policy, qcPre the stack of pretrained policies and lambda their
%   weights (vlambda holds the same weights in log space).
%
%   See also NAIVE_INIT, LEARNER_INIT, TRANSFER_LEARNING.
Q.t=1; % trial

% world
Q.s = zeros(length(D),T); % hidden states
Q.o = zeros(length(qaInit(:,1)), T); % outcome
Q.d= zeros(length(E),T); % decisions
Q.u = zeros(4,T); % action
Q.G = ones(1,T); % risk

% beliefs
Q.qs = zeros(length(D),T); % posterior s
Q.qd = zeros(length(E),T); % posterior d
Q.qd_ = zeros(length(E),T); % normalized posterior d
Q.vs = zeros(length(D),T); % v of qs
Q.vd = zeros(length(E),T); % v of qd
Q.qs(:,1) = D;
Q.qd(:,1) = E;

Q.qa = qaInit;
Q.qb = qbInit;
Q.qc1 = qc1Init;
Q.qcPre = qcPreInit; % 10000x4x(length(lambda)-1)
Q.lambda = lambdaInit;
Q.vlambda = lambdaInit;
Q.D = D;
Q.E = E;

end