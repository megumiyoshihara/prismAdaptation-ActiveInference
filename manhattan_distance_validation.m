function distance = manhattan_distance_validation(Nm, s, targetX, targetY)
%MANHATTAN_DISTANCE_VALIDATION  Grid steps from the hand to a fixed target.
%   DISTANCE = MANHATTAN_DISTANCE_VALIDATION(NM,S,TARGETX,TARGETY) is the
%   validation-side counterpart of MANHATTAN_DISTANCE.  There the state S encodes
%   the hand only, as (handX-1)*Nm + handY, so the target comes in as arguments.
%
%   See also MANHATTAN_DISTANCE.
sPos = find(s==1)-1;
pos1_1 = floor(sPos/Nm)+1;
pos1_2 = mod(sPos, Nm)+1;


distance = abs(pos1_1-targetX) + abs(pos1_2-targetY);
end