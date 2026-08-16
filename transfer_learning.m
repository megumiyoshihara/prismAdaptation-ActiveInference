function Q = transfer_learning(Q0)
%TRANSFER_LEARNING  One session of Dirichlet updates for the transfer learner.
%   Q = TRANSFER_LEARNING(Q0) accumulates qa and qb exactly as
%   NAIVE_LEARNING does, but instead of moving a policy it moves the mixture
%   weights: each pretrained policy collects the risk-weighted evidence
%   (1-2*Gamma)*lnC*qs' * d, which is added to log(lambda) and softmaxed back
%   into Q.lambda (floored at 1e-4).
%
%   The addNewC switch at the top keeps the learner's own policy qc1 in the
%   mixture as well; it is 0, so only the pretrained policies are weighted.
%
%   See also NAIVE_LEARNING, LEARNER_UPDATE, PARAM_NORMALIZATION.
    addNewC = 0;
    Q = Q0;
    Q.t = 1;
    Ns = length(Q0.s(:,1));
    Nu = length(Q0.u(:,1));
    t = Q0.t;

    Q.qa = Q0.qa + Q0.o(:,2:t) * Q0.qs(:, 2:t)';

    Q.qb = Q0.qb + Q0.qs(:, 2:t)*(kron(Q0.u(:, 1:t-1), ones(Ns, 1)).*kron(ones(Nu,1),Q0.qs(:, 1:t-1)))';

    qs = Q0.qs(:, 1:t-1);
    d = Q0.d(:, 2:t);
    

    % Cat distribution
    %%{
    lambda = Q.lambda;
    lambdaSum = sum(lambda);
    Lambda = log(lambda/lambdaSum);
    vlambda = zeros(size(lambda));
    lambdaT = length(lambda);
    if addNewC == 1
        for i=1:t-2
            tt = i;
            G = Q.Gamma(i+2);
            [~, lnC] = param_normalization(Q0.qc1, "A");
            vlambda(1) = vlambda(1)+((1-2*G)*lnC*qs(:,tt))'*d(:,tt);
        
            for j=2:lambdaT
                [~, preqlnC] = param_normalization(Q0.qcPre(:,:,j-1), "A");
                vlambda(j) = vlambda(j)+((1-2*G)*preqlnC*qs(:,tt))'*d(:,tt);
            end

            Q.qc1 = max(Q.qc1+(1-2*G)*d(:,tt)*qs(:,tt)',0.1);
        end
    elseif addNewC == 0
        for i=1:t-2
            tt = i;
            G = Q.Gamma(i+2);
        
            for j=1:lambdaT
                [~, preqlnC] = param_normalization(Q0.qcPre(:,:,j), "A");
                vlambda(j) = vlambda(j)+((1-2*G)*preqlnC*qs(:,tt))'*d(:,tt);
            end

        end
    end

    %%}


    % forget post experience
    vlambda = Lambda + vlambda;
    Q.lambda = max(exp(vlambda-max(vlambda))./sum(exp(vlambda-max(vlambda))), 0.0001);
    Q.vlambda = vlambda;
end