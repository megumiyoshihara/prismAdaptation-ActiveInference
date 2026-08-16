function s = calc_xy2S(Nm, targetX, targetY, handX, handY)
%CALC_XY2S  One-hot state vector for a hand and target position pair.
%   Calculate a one-hot state vector for the pair of hand and target positions based on the coordinates.
%   S = CALC_XY2S(NM,TARGETX,TARGETY,HANDX,HANDY) returns an Nm^4 x 1 indicator
%   of the packed state index
%       s = (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX-1)*Nm + targetY
%
%   See also MAKE_RANDHANDS, MANHATTAN_DISTANCE.
s=zeros(Nm^2*Nm^2, 1);
s((handX-1)*Nm^3+(handY-1)*Nm^2+(targetX-1)*Nm+targetY,1)=1;
end