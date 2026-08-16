function distance = manhattan_distance(Nm, s)
%MANHATTAN_DISTANCE  Grid steps between the hand and the target.
%   DISTANCE = MANHATTAN_DISTANCE(NM,S) unpacks the one-hot state S, whose index
%   holds both positions as
%       s = (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX-1)*Nm + targetY
%   and returns |dx| + |dy| between them.
%
%   See also MANHATTAN_DISTANCE_VALIDATION, CALC_XY2S.
sPos = find(s==1)-1;
pos1 = floor(sPos/Nm^2);
pos2 = mod(sPos,Nm^2);
pos1_1 = floor(pos1/Nm)+1;
pos1_2 = mod(pos1, Nm)+1;
pos2_1 = floor(pos2/Nm)+1;
pos2_2 = mod(pos2, Nm)+1;


distance = abs(pos1_1-pos2_1) + abs(pos1_2-pos2_2);
end