function [EeeT,SNR,rho2,fMSE] = TC(tri)
%% This function is a classic version of TC results
% LCM (licm_13@163.com) have a good day!
%
% INPUT
%   tri = inputs (nx3)
%
% OUTPUT 
%   EeeT = error covariance matrix (3x3) zero ecc
%   SNR = signal to noise ratio (3x1, = beta2_sigTheta2/sige2)
%   rho2 = data-truth correlation (3x1, = SNR/(1+SNR))
%   fMSE = fractional mean-square-error (3x1, = 1-rho2)
% NOTES (details refer to doc file)
%   y = Ax : x = ((A'A)^-1)A'y
%   y = known variance combinations (6x1)
%   A = equation matrix (6x6)
%   x = TC results (6x1)
%   x(1:3) = SNR
%   x(4:6) = EeeT
if size(tri,1)<100
    EeeT = 0.01*eye(3); SNR = [0;0;0]; rho2 = SNR; fMSE = SNR;
else
    ExxT = cov(tri);
    A = [1,0,0,1,0,0;...
         0,1,0,0,1,0;...
         0,0,1,0,0,1;...
         1,0,0,0,0,0;...
         0,1,0,0,0,0;...
         0,0,1,0,0,0];
    y = [diag(ExxT);...
         ExxT(1,2)*ExxT(1,3)/ExxT(2,3);...
         ExxT(2,1)*ExxT(2,3)/ExxT(1,3);...
         ExxT(3,1)*ExxT(3,2)/ExxT(1,2)];
    x = ((A'*A)^-1)*A'*y;
    EeeT = diag(x(4:end));
    SNR = x(1:3)./x(4:end);
    rho2 = SNR./(1+SNR);
    fMSE = 1 - rho2;
end
end