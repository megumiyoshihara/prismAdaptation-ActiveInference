function Q = naive_init(qa0,qb0,qc0,D,E,T)
%NAIVE_INIT  Trial buffers and priors for one session of the naive learner.
%   Q = NAIVE_INIT(QA0,QB0,QC0,D,E,T) allocates the world traces (s, o, d, u, G)
%   and the posteriors (qs, qd, qd_, vs, vd) for T trials, seeds trial 1 with the
%   priors D and E, and carries the Dirichlet parameters qa, qb, qc.  The naive
%   learner keeps its policy in qc itself.
%
%   See also TRANSFER_INIT, LEARNER_INIT, NAIVE_LEARNING.
Q.t=1; % trial

% world
Q.s = zeros(length(D),T); % hidden states
Q.o = zeros(length(qa0(:,1)), T); % outcome
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

Q.qa = qa0;
Q.qb = qb0;
Q.qc = qc0;
Q.D = D;
Q.E = E;

end