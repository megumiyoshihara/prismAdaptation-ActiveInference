function optC = make_optC(Nm, perceivedX, perceivedY, targetX, targetY)
%MAKE_OPTC  Optimal goal prior C for the prism exposure period.
%   OPTC = MAKE_OPTC(NM, PERCEIVEDX, PERCEIVEDY, TARGETX, TARGETY) returns the
%   4 x Nm^4 reference policy the generalisation error is measured against.
%
%   While the prism is on, the agent perceives the target at
%   (PERCEIVEDX, PERCEIVEDY) but the hand has to reach (TARGETX, TARGETY), so
%   the reference lives in the state columns of the perceived target while its
%   movement directions point at the real one.
%
%   C maps the state one step earlier to the action (see the qlnC * qs(:,t-1)
%   term in MAIN_LEARNER), so the naive per-position optimum is propagated one
%   step back through the hand transition matrices before it is normalised.
%
%   If the perceived position is (7,5) and the actual position is (4,5), the result
%   generated matches exactly with the one used in the paper.
%
%   See also REPEAT_MAIN, MAKE_BHAND.

arguments
    Nm         (1,1) double
    perceivedX (1,1) double
    perceivedY (1,1) double
    targetX    (1,1) double
    targetY    (1,1) double
end

% Naive optimum: uniform over whichever of [up down left right] reduces the
% Manhattan distance to the real target, uniform over all four once on it.
optsc = zeros(4, Nm^2);
for handX = 1:Nm
    for handY = 1:Nm
        k = (handX-1)*Nm + handY;
        reduces = [targetY < handY;    % up
                   targetY > handY;    % down
                   targetX < handX;    % left
                   targetX > handX];   % right
        if any(reduces)
            optsc(:,k) = reduces / sum(reduces);
        else
            optsc(:,k) = 0.25;
        end
    end
end

% Evaluate that policy from the state one step earlier: mark each action as
% risky or not, then push it back through every possible preceding move.
B     = make_Bhand(Nm);
Bsum  = B(:,1:Nm^2) + B(:,Nm^2+1:2*Nm^2) + B(:,2*Nm^2+1:3*Nm^2) + B(:,3*Nm^2+1:4*Nm^2);
prot  = max(sign(optsc - 0.3) * Bsum, 0);
prot(:, (targetX-1)*Nm + targetY) = 0.25;
prot  = double(prot ./ sum(prot, 1));

% Place the block in the state columns belonging to the perceived target.
optC = zeros(4, Nm^4);
optC(:, ((perceivedX-1)*Nm + perceivedY):Nm^2:Nm^4) = prot;
end
