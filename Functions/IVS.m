function [RMSE, rho2] = IVS(X,N_boot,column,M_A)
%% Multiplicative %[RMSE, rho2] = IVS(X,N_boot,column)
% rho is averaged CC^2 result
% N_R & N_C are the ^2 results 
% column is the chosen one to be used for generating Lag-1 series

if size(X,1)<4
    RMSE = [-9999 -9999];
    rho2 = [-9999 -9999];
else
    
    % check input data
    if size(X,2) ~= 2 
        error('Error: Input data X must be an N x 2 matrix');
    end

    if any(isnan(X(:)))
        error('Error: Input data X must not contain NaNs');
    end
    
    % creating empty outputs
    RMSE = zeros(1,2);
    rho2 = zeros(1,2);
    
    if M_A == 1 % default multiplicative        
        % dealing with minus value
        disp('Multiplicative mode');
        X(X<0) = 0.01;
            
        % Log transformation of data
        X = log(X);
    elseif M_A == 0 % additive
        disp('Additive mode');
    else
        error('Error: M_A wrong input, 1 for *;0 for +');
    end
    
    % add Lag-1 series
    if column == 1
        I = X(2:length(X),1);
    elseif column == 2
        I = X(2:length(X),2);
    else
        error('Error: column wrong input, 1 or 2');
    end
    
    % deleting the last line & add I
    X(length(X),:) = [];
    X(:,3) = I;
            
    if isempty(X) == 0
        
        % opts = statset('UseParallel', true);
        % Number of samples for Bootstrap
        
        % creating empty temporal storage
        N_R = zeros(N_boot,2);
        N_C = zeros(N_boot,2);
        N_s_ivs = zeros(N_boot,1);
        
        % Bootstrp
        boot = bootstrp(N_boot,'nancov',X);
        
        for i = 1:N_boot
            
            % creating empty temporal storage for R2 and CC
            R2 = zeros(1,2);
            CC = zeros(1,2);
            
            % reshape the Q to 3*3 matrix
            Q = reshape(boot(i,:),3,3);
            
            % calculating scaling ratio
            s_ivs = abs(Q(3,1) / Q(3,2));
            
            % calcuating R2
            R2(1) = Q(1,1) - Q(1,2) * s_ivs;
            R2(2) = Q(2,2) - Q(1,2) / s_ivs;
            % dealing with minus R2 value and transfer to R
            R2(R2<0) = 0.0001; % 加了这么一条判断用的语句，不知道行不行
            R = sqrt(R2);
            R(R<10^-3) = 0.0001;
            
            % calculating CC
            CC(1) = Q(1,2) * s_ivs / Q(1,1);
            CC(2) = Q(1,2) / ( Q(1,1) * s_ivs );
            
            % dealing with minus CC value and transfer to C
            CC(CC<0) = 0.0001; % 加了这么一条判断用的语句，不知道行不行
            CC(CC>1) = 0.9999;
                        
            N_R(i,:) = R;
            N_C(i,:) = CC;
            N_s_ivs(i,:) = s_ivs;
            
        end
        RMSE(1) = mean(N_R(:,1));
        RMSE(2) = mean(N_R(:,2));
        
        rho2(1) = mean(N_C(:,1));
        rho2(2) = mean(N_C(:,2));
    else
        RMSE = [-9999 -9999];
        rho2 = [-9999 -9999];
    end
end
end
