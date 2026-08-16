function Q = naive_learning(Q0)
%NAIVE_LEARNING  One session of Dirichlet updates for the naive learner.
%   Q = NAIVE_LEARNING(Q0) accumulates the likelihood counts qa and the
%   transition counts qb from the trial traces of Q0, then walks the policy qc
%   trial by trial with the risk-weighted outer product (1-2*Gamma)*d*qs',
%   floored at 0.1.  The loop advances one trial at a time.
%
%   See also TRANSFER_LEARNING, LEARNER_UPDATE, CALC_RISK.
    Q = Q0;
    Q.t = 1;
    Ns = length(Q0.s(:,1));

    Nu = length(Q0.u(:,1));
    t = Q0.t;

    Q.qa = Q0.qa + Q0.o(:,2:t) * Q0.qs(:, 2:t)';
    
    Q.qb = Q0.qb + Q0.qs(:, 2:t)*(kron(Q0.u(:, 1:t-1), ones(Ns, 1)).*kron(ones(Nu,1),Q0.qs(:, 1:t-1)))';

    qs = Q0.qs(:, 1:t-1);
    d = Q0.d(:, 2:t);

    for i=1:t-2
        tt = i;
        G = Q.Gamma(i+2);
        Q.qc = max(Q.qc+(1-2*G)*d(:,tt)*qs(:,tt)',0.1);
    end
end