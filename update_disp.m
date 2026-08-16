function update_disp(msg_pre,msg,endDisp)
%UPDATE_DISP  Overwrite the previous progress message in place.
%   This is code for displaying the progress of the simulation.
%   UPDATE_DISP(MSG_PRE,MSG,ENDDISP) erases MSG_PRE with backspaces and prints
%   MSG on the same line, so a batch driver can show a progress counter without
%   scrolling the command window.  Pass ENDDISP = 1 on the last call to end the
%   line with a newline.
%
%   The caller must pass the exact string it printed last as MSG_PRE; anything
%   printed in between (a warning, a figure message) breaks the alignment.
%
%   See also REPEAT_MAIN, REPEAT_VALIDATION.
    len = length(msg_pre);
    fprintf(repmat('\b', 1,len))
    fprintf('%s',msg);
    drawnow;
    if endDisp==1
        fprintf('\n'); 
    end
end