function figure_output_validation(Nm, s, tX, tY, c)
%FIGURE_OUTPUT_VALIDATION  Draw one frame of the 3x3 validation experiment.
%   FIGURE_OUTPUT_VALIDATION(NM,S,TX,TY,C) draws the policy C as a colour per
%   cell, the target in white and the hand in black, the way
%   FIGURE_OUTPUT_MAIN does for the main experiment.  The state S indexes the
%   hand position only, so it is reshaped to the Nm x Nm field directly and the
%   target comes in as coordinates rather than being decoded out of the state.
%
%   FIGURE_OUTPUT_VALIDATION(NM,S,TX,TY) leaves the policy out and shows the
%   hand alone, with the target overlaid at half intensity.  Mixing the colours
%   costs far more than the rest of the drawing, so this is the form to use
%   where the policy is not the point, as in the pretraining.
%
%   See also FIGURE_OUTPUT_MAIN, CFORDISPLAY_VALIDATION, VALIDATION_LEARNER.

hand = reshape(s, Nm, Nm);
target = zeros(Nm,Nm);
target(tY,tX) = 1;

if nargin < 5 || isempty(c)
    display = hand + 0.5*target;
    image(display,'CDataMapping','scaled');
    axis square;
    return;
end

% Set color in colormap
colorUp = hsv2rgb([0,1,1]);
colorRight = hsv2rgb([0.25,1,1]);
colorLeft = hsv2rgb([0.75,1,1]);
colorDown = hsv2rgb([0.5,1,1]);

colors = [colorUp;colorDown;colorLeft;colorRight];
normC = param_normalization(c, "A");
matrixC = reshape(normC' * colors, Nm, Nm, 3);
hsvmatrixC = rgb2hsv(matrixC);
% change Saturation
hsvmatrixC(:,:,2)= min(hsvmatrixC(:,:,2)*1.3,1) ;

rgbmatrixC = hsv2rgb(hsvmatrixC)*255;

RGB_C = uint8(round(rgbmatrixC));
hand(:,:,2) = hand;
hand(:,:,3) = hand(:,:,1);

target(:,:,2) = target;
target(:,:,3) = target(:,:,1);

display = RGB_C;
display(target==1) = 255;
display(hand==1) = 0;
image(display);
axis square;
txt = '{\color[rgb]{1, 0, 0}■\color[rgb]{0,0,0}up \color[rgb]{0,1,1}■\color[rgb]{0,0,0}down \color[rgb]{0.5,0,1}■\color[rgb]{0,0,0}left \color[rgb]{0.5,1,0}■\color[rgb]{0,0,0}right}';
subtitle(txt);
fontsize(gca,30,"pixels");
end
