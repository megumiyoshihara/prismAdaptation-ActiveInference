function B = make_Bhand(size)
%MAKE_BHAND  Transition matrix of the hand on the monitor grid.
%   B = MAKE_BHAND(SIZE) returns the Nm^2 x (4*Nm^2) matrix, Nm = SIZE, that
%   maps kron(delta(t),s_h(t)) to s_h(t+1): the hand moves one cell in the
%   direction the action selects and stays put when that would leave the grid.
%
%   The four Nm^2 x Nm^2 blocks are concatenated in the order
%   [up, down, left, right], matching the action index used everywhere else.
%   Each block is a Kronecker product of the identity with the one-step shift M
%   (or its flip), because the x and the y coordinate move independently.
%
%   ex. size=4 -> M=[1,1,0,0;0,0,1,0;0,0,0,1;0,0,0,0]
%   The doubled first column is what makes the hand stay on the border.
%
%   See also CALC_DXS, MAKE_ATARGETSHIFT.
M = [eye(size,1, 'single'),[eye(size-1, 'single');zeros(1,size-1, 'single')]];

I = eye(size, 'single');
B_up =kron(I, M);
B_down =kron(I, flip(flip(M,1),2));
B_left =kron(M, I);
B_right =kron(flip(flip(M,1),2),I);
B=[B_up,B_down,B_left,B_right];
end