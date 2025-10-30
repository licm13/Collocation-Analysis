function [EeeT,SNR,rho2,fMSE,L] = EIVD(tri)
%% This function is to calculate EIVD results
% Attention
%   for a [x y z] input only (y,z) have non-zero ecc
% INPUT
%   tri = inputs (nx3) no need for rescaling
%
% OUTPUT 
%   EeeT = error covariance matrix (3x3)
%   SNR = signal to noise ratio (3x1, = beta2_sigTheta2/sige2)
%   rho2 = data-truth correlation (3x1, = SNR/(1+SNR))
%   fMSE = fractional mean-square-error (3x1, = 1-rho2)
%   ecc = non-zero error cross-correlation <y,z> (unitless)
%   L = Lag-1 [tem-res] series
% NOTES (details refer to doc file)
%   y = Ax : x = ((A'A)^-1)A'y
%   y = known variance combinations (10x1)
%   A = equation matrix (10x8)
%   x = EIVD results (8x1)

     % Calculate error covariance matrix
    ExxT = cov(tri);

    % Generate Lag-1 [tem-res] series
    tri_lag = tri(2:end,:);
    tri(end,:) = [];
    L = mean(tri.*tri_lag);

    % Calculate EIVD results
    A = [1,0,0,0,1,0,0,0;...
         0,1,0,0,0,1,0,0;...
         0,0,1,0,0,0,1,0;...
         0,0,0,1,0,0,0,1;...
         1,0,0,0,0,0,0,0;...
         1,0,0,0,0,0,0,0;...
         0,1,0,0,0,0,0,0;...
         0,0,1,0,0,0,0,0;...
         0,0,0,1,0,0,0,0;...
         0,0,0,1,0,0,0,0];
         
    y = [diag(ExxT);ExxT(2,3);...
         ExxT(1,2)*sqrt(L(1)/L(2));...
         ExxT(1,3)*sqrt(L(1)/L(3));...
         ExxT(2,1)*sqrt(L(2)/L(1));...
         ExxT(3,1)*sqrt(L(3)/L(1));...
         ExxT(1,2)*sqrt(L(3)/L(1));...
         ExxT(1,3)*sqrt(L(2)/L(1))];
    x = ((A'*A)^-1)*A'*y;

    % assemble error variance matrix
    EeeT = diag(x(5:7));
    EeeT(2,3) = x(end); EeeT(3,2) = x(end);

    SNR = x(1:3)./x(5:7);
    rho2 = SNR./(1+SNR);
    fMSE = 1 - rho2;
end