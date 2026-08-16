function Gamma = calc_risk(Q,range, del_th,risk)
%CALC_RISK  Per-trial risk from the trajectory of the session.
%   GAMMA = CALC_RISK(Q,RANGE,DEL_TH,RISK) scores every trial up to Q.t: it maps
%   del = (1 - 2*Q.G(t))*RANGE onto the partition DEL_TH and picks the matching
%   level out of RISK.  Trials past Q.t stay zero.
%
%   DEL_TH holds the thresholds in ascending order, and RISK one level per cell
%   of the partition they cut - the thresholds counting as cells of their own,
%   so N thresholds need 2*N+1 levels, ordered
%
%      below th(1), at th(1), between th(1) and th(2), at th(2), ...,
%      between th(N-1) and th(N), at th(N), above th(N)
%
%   The simulations use DEL_TH = [-1, 0, 1] with seven levels, but any number of
%   thresholds works as long as RISK is that much longer.
%
%   See also NAIVE_LEARNING, TRANSFER_LEARNING.

Nth = length(del_th);
if length(risk) ~= 2*Nth + 1
    error("calc_risk:riskLength", ...
        "risk needs 2*length(del_th)+1 = %d levels, but %d were given.", ...
        2*Nth + 1, length(risk));
end

T = length(Q.o(1,:));
t = Q.t;

Gamma = zeros(T, 1);

for i = 1:t
    tt = i;
    del = 1- mean(Q.G(tt)) * 2;

    del = del * range;
    % Walk the thresholds upwards and stop at the first one del does not clear;
    % del above them all keeps the last level.
    level = 2*Nth + 1;
    for k = 1:Nth
        if del < del_th(k)
            level = 2*k - 1;
            break;
        elseif del == del_th(k)
            level = 2*k;
            break;
        end
    end
    Gamma(i) = risk(level);
end
end
