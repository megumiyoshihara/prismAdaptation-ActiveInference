function [qc, qlnC] = learner_policy(cfg, Qh)
%LEARNER_POLICY  Goal prior C the learner acts on in the current session.
%   [QC, QLNC] = LEARNER_POLICY(CFG, QH) returns the normalised C used for
%   display (QC) and the log-scale C that enters the decision value Q.vd (QLNC).
%
%     "naive"    QLNC is the expected log of the Dirichlet posterior over C,
%                straight out of PARAM_NORMALIZATION.
%     "transfer" QLNC is the lambda-weighted sum of the log pretrained
%                policies, left unnormalised; QC is its softmax.
%
%   This is the only place where the two learners' decision-making differs.
%
%   See also MAIN_LEARNER, LEARNER_INIT, LEARNER_UPDATE.

switch cfg.sim_type
    case "naive"
        [qc, qlnC] = param_normalization(Qh.qc, "A");
    case "transfer"
        qlnC = zeros(size(Qh.qc1));
        for i = 1:numel(Qh.lambda)
            preqc = param_normalization(Qh.qcPre(:,:,i), "A");
            qlnC  = qlnC + Qh.lambda(i) * log(preqc);
        end
        qc = exp(qlnC) ./ sum(exp(qlnC));
    otherwise
        error("learner_policy:unknownSimType", ...
              "sim_type must be ""naive"" or ""transfer"", got ""%s"".", cfg.sim_type);
end
end
