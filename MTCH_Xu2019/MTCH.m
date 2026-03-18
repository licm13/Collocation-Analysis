function [std] = MTCH(Z)
% The Multiplicative-error-based Three-Cornered Hat (MTCH) method calculates the root-mean-squared-
% error (RMSE) between three or more sets of data (from 
% different measurement systems) and the true.
% 
%The difference between MTC Hand classical TCH is that in MTCH the error model 
% is multiplicative which is more appropriate in case of e.g. evapotranspiration data.
%
%%% Input:
%
%       Z    is the data matrix with M rows and N columns 
%       (M is the total number of time samples in each product).
%
%%% Outputs:    
%
%       std is a 1 * N vector representing the RMSE between each of the data 
%            sets and the true.
%
%
%
%%% Reference
%
% Li X.X., . 
% (2015), Improving global evapotranspiration estimation by merging 
% four products based on multiplicative error model and the three-cornered hat method,  
%
%
%
% Version: 15, November 2025.
% Author:  X. Li, katelisomeone@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R=[];
%X=[x1 x2 x3 x4];
X = log(Z);
[M,N]=size(X);
%Reference data
ture_num=N;
Y=[];

for i=1:N
    if i~=ture_num
        y=X(:,i)-X(:,ture_num);
        Y=[Y,y];
    end
end
S = cov(Y);
u=[];
for i=1:N-1
    u(i)=1;
end
%Set the initial conditions of the iteration
for i=1:length(S)
    R(i,N)=0;
end
R(N,N) = (2*u*inv(S)*u')^-1;    %inv(): Inverse matrix
x0 = R(:,N);
%According to the initial conditions, constraint conditions, and objective function of the iteration, R(:,N) is calculated
opts = optimset('Algorithm','active-set','TolX',2e-10,'TolCon',2e-10,'Display','off');
x= fmincon(@(r) myfun(r,S),x0,[],[],[],[],[],[],@(r) mycon(r,S),opts);
%Using the relationship between S and R and the R(:,N) solved in the previous step, find the remaining unknowns in the R matrix
for i=1:N
    R(i,N) = x(i);
end
for i=1:length(S)
    for j=1:length(S)
        if i<j | i==j
            R(i,j)= S(i,j)-R(N,N) +R(i,N)+R(j,N);
        end
    end
end
for i=1:length(S)+1
    for j=1:length(S)+1
        R(j,i)=R(i,j);
    end
end
std_log=[];
for i=1:N
    st=sqrt(R(i,i));
    std_log=[std_log st];
end
std = std_log.*nanmean(Z)