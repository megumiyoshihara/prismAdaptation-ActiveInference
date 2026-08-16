function [Qnext, transientFields] = learner_update(cfg, Qh)
%LEARNER_UPDATE  Apply one session of learning and report the fields to drop.
%   [QNEXT, TRANSIENTFIELDS] = LEARNER_UPDATE(CFG, QH) returns the posterior for
%   the next session and the list of large per-trial fields that MAIN_LEARNER
%   strips from the stored session snapshot to keep memory use down.
%
%   The transfer learner additionally drops qcPre and qc1: the pretrained stack
%   is identical in every session, and qc1 is not updated (see TRANSFER_LEARNING),
%   so the effective policy is reconstructed downstream from lambda instead.
%
%   See also MAIN_LEARNER, LEARNER_INIT, LEARNER_POLICY.

switch cfg.sim_type
    case "naive"
        Qnext           = naive_learning(Qh);
        transientFields = {'qa', 'qb', 'qs', 'vs', 'D', 'E'};
    case "transfer"
        Qnext           = transfer_learning(Qh);
        transientFields = {'qa', 'qb', 'qs', 'vs', 'E', 'D', 'qcPre', 'qc1'};
    otherwise
        error("learner_update:unknownSimType", ...
              "sim_type must be ""naive"" or ""transfer"", got ""%s"".", cfg.sim_type);
end
end
