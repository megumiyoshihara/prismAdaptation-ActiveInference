function kronecker = calc_DxS(u,s)
%CALC_DXS  Kronecker product of a one-hot action and a state vector.
%   KRONECKER = CALC_DXS(U,S) returns kron(delta(U),S), the (Nu*Ns) x 1 vector
%   the transition matrix B is applied to.  U is the one-hot action (Nu x 1) and
%   S the state (Ns x 1); the result is S placed in the block selected by U and
%   zero everywhere else.
%
%   Written out rather than calling kron() because only one block is non-zero.
%
%   See also MAKE_BHAND, CALC_XY2S.
Nu = size(u,1);
Ns = size(s,1);
length = Nu*Ns;
kronecker = zeros(length,1);
vec = 0:Nu-1;
kronecker((vec*u)*Ns+1:(vec*u+1)*Ns,:)=s;
end