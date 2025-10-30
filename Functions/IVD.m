function [EeeT, rho2,u] = IVD(dual)
%% This function is to calculate IVD results considering the choice of offset
% Attention
%   arbitrary offsets tend to produce unstable results
%       R_Ix * R_Jy > 0
%   ensures the same trend of changes between the instruments
%   prevents the negative correlation of random errors
%       offset \belong argmax(R_Ix+r_Jy)
%   the selected offset makes the given expression the sum of R_Ix and R_Jx
%   attains its maximum value
% INPUT
%   dual = inputs (nx2)
% OUTPUT 
%   EeeT = error covariance matrix (2x2)
%   rho2 = data-truth correlation (2x1)
%   u = linear-squared merging weights (2x1)
% Ref
%   FUSING ACTIVE AND PASSIVE REMOTELY SENSED SOIL MOISTURE PRODUCTS
%   USING INFORMATION VECTORS
%   https://www.researchgate.net/publication/228844200_FUSING_ACTIVE_AND_PASSIVE_REMOTELY_SENSED_SOIL_MOISTURE_PRODUCTS_USING_INFORMATION_VECTORS
if size(dual,1)<4
    EeeT = nan(2,2);
    rho2 = nan(1,2); u = nan(1,2);
else
    
    % check input data
    if size(dual,2) ~= 2 
        error('Error: Input data X must be an N x 2 matrix');
    end

    if any(isnan(dual(:)))
        error('Error: Input data X must not contain NaNs');
    end
    
    X = dual(:,1); Y = dual(:,2);
    % find offset
    sumR = 0; offset = 0;
    for i = 1:length(X)
        tri_I = X(1+i:end); tri_J = Y(1+i:end);
        tri_X = X; tri_X(end-i+1:end) = [];
        tri_Y = Y; tri_Y(end-i+1:end) = [];
        judge = corrcoef(tri_X,tri_I) + corrcoef(tri_Y,tri_J);
        judge = max(max(judge));
        if judge>sumR
            sumR = judge;
            offset = i;
        end
    end
    % Generate Lag-offset series 
    I = X(1+offset:end); J = Y(1+offset:end);
    X(end-offset+1:end) = []; Y(end-offset+1:end) = [];
    
    % Calculate IVD variances
    combin = [X I Y J];    ExxT = cov(combin);
    u = 0; rho2 = 0; EeeT = zeros(2,2);
    s_ivd = sqrt(ExxT(1,2)/ExxT(3,4));
    EeeT(1,1) = ExxT(1,1) - ExxT(1,2) * s_ivd;
    EeeT(2,2) = ExxT(2,2) - ExxT(2,1) / s_ivd;
    u(1) = (EeeT(2,2)*s_ivd)/(EeeT(1,1)+EeeT(2,2)*s_ivd);
    u(2) = (EeeT(1,1)*s_ivd)/(EeeT(1,1)+EeeT(2,2)*s_ivd);
    rho2(1) = ExxT(1,1) - ExxT(1,3) * s_ivd;
    rho2(2) = ExxT(2,2) - ExxT(1,3) / s_ivd;
    
end
end
