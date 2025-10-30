function [r,KGE] = KGE_ObjFun(Rsim,Robs)
%----------------------------the following is the objective function of KGE, i.e.,Kling-Gutpa Efficency, ...
% you can also use the other objective fucntion, such as NSE and PBIAS.For KGE,please refer the following paper:
% Gupta, H. V.etal. (2009). Decomposition of the mean squared error and NSE 
% performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1), 80-91.
observed=Robs;
modelled=Rsim;
% observed=Pyear-Ryear;modelled=Ea_year;
% cflow=[modelled,observed];
sdmodelled=std(modelled);
sdobserved=std(observed); 
mmodelled=mean(modelled);
mobserved=mean(observed);
r=corrcoef(observed,modelled);
r=r(1,2);
relvar=sdmodelled/sdobserved;
bias=mmodelled/mobserved;
% %KGE timeseries 
KGE=sqrt( ((r-1)^2) + ((relvar-1)^2)  + ((bias-1)^2) );
% KGE=sqrt( 0.8*((r-1)^2) + 0.2*((bias-1)^2));
end