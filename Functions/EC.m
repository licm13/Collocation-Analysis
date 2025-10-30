function results = EC(qu)
% EC - Extended Collocation results for all possible combinations of coordinates.
%
% INPUT:
%   qu: Input data (nx4) containing coordinates 'a', 'b', 'c', and 'd'.
%
% OUTPUT:
%   results: Struct array containing EC results for each combination.

% Initialize results struct array to store the EC results
results = struct('EeeT', [], 'SNR', [], 'rho2', [], 'fMSE', [], 're_weight', [], 'res_qu', [], 'weighted_result', []);

% Define all possible combinations of coordinates
combinations = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 3, 4];

% Check if the input 'qu' contains exactly 4 columns
if size(qu, 2) == 4
    % Loop through each combination
    for comb_idx = 1:size(combinations, 1)
        ref_idx = combinations(comb_idx, :);

        % Permute columns of 'qu' based on the current combination
        perm_qu = qu(:, [ref_idx(1), ref_idx(2), setdiff(1:4, ref_idx)]);
        
        % Analysis the rows with <0 or nan data and collect the row_numbers
        invalid_rows = any(perm_qu < 0, 2) | any(isnan(perm_qu), 2);
        valid_rows = ~invalid_rows;
        cal_qu = perm_qu(valid_rows, :);
        % Check if the input contains at least 4 rows
        if size(cal_qu, 1) < 4
            results = [];
            return;
        end
        
        % Initialize 'res_qu' to store the results of the permutation
        res_qu = zeros(size(cal_qu, 1), size(cal_qu, 2), 4);
        ExxT_unres = cov(cal_qu);

        % Loop through each reference coordinate 'a', 'b', 'c', and 'd'
        for i = 1:4
            if i == 1
                % Reference: 'a'
                res_qu(:, 1, i) = cal_qu(:, 1);
                res_qu(:, 2, i) = (ExxT_unres(1, 3) / ExxT_unres(2, 3)) * (cal_qu(:, 2) - nanmean(cal_qu(:, 2))) + nanmean(cal_qu(:, 1));
                res_qu(:, 3, i) = (ExxT_unres(1, 2) / ExxT_unres(3, 2)) * (cal_qu(:, 3) - nanmean(cal_qu(:, 3))) + nanmean(cal_qu(:, 1));
                res_qu(:, 4, i) = (ExxT_unres(1, 2) / ExxT_unres(4, 2)) * (cal_qu(:, 4) - nanmean(cal_qu(:, 4))) + nanmean(cal_qu(:, 1));
            elseif i == 2
                % Reference: 'b'
                res_qu(:, 1, i) = (ExxT_unres(2, 3) / ExxT_unres(1, 3)) * (cal_qu(:, 1) - nanmean(cal_qu(:, 1))) + nanmean(cal_qu(:, 2));
                res_qu(:, 2, i) = cal_qu(:, 2);
                res_qu(:, 3, i) = (ExxT_unres(2, 1) / ExxT_unres(3, 1)) * (cal_qu(:, 3) - nanmean(cal_qu(:, 3))) + nanmean(cal_qu(:, 2));
                res_qu(:, 4, i) = (ExxT_unres(2, 1) / ExxT_unres(4, 1)) * (cal_qu(:, 4) - nanmean(cal_qu(:, 4))) + nanmean(cal_qu(:, 2));
            elseif i == 3
                % Reference: 'c'
                res_qu(:, 1, i) = (ExxT_unres(3, 2) / ExxT_unres(1, 2)) * (cal_qu(:, 1) - nanmean(cal_qu(:, 1))) + nanmean(cal_qu(:, 3));
                res_qu(:, 2, i) = (ExxT_unres(3, 1) / ExxT_unres(2, 1)) * (cal_qu(:, 2) - nanmean(cal_qu(:, 2))) + nanmean(cal_qu(:, 3));
                res_qu(:, 3, i) = cal_qu(:, 3);
                res_qu(:, 4, i) = (ExxT_unres(3, 1) / ExxT_unres(1, 4)) * (cal_qu(:, 4) - nanmean(cal_qu(:, 4))) + nanmean(cal_qu(:, 3));
            elseif i == 4
                % Reference: 'd'
                res_qu(:, 1, i) = (ExxT_unres(4, 2) / ExxT_unres(1, 2)) * (cal_qu(:, 1) - nanmean(cal_qu(:, 1))) + nanmean(cal_qu(:, 4));
                res_qu(:, 2, i) = (ExxT_unres(4, 1) / ExxT_unres(2, 1)) * (cal_qu(:, 2) - nanmean(cal_qu(:, 2))) + nanmean(cal_qu(:, 4));
                res_qu(:, 3, i) = (ExxT_unres(4, 1) / ExxT_unres(3, 1)) * (cal_qu(:, 3) - nanmean(cal_qu(:, 3))) + nanmean(cal_qu(:, 4));
                res_qu(:, 4, i) = cal_qu(:, 4);
            end
        end
        % Store the results of the permutation in the results struct array
        results(comb_idx).res_qu = res_qu;

        % Loop through each permutation and calculate EC results
        for i = 1:3
            % Calculate the rescaled error covariance matrix
            ExxT = cov(res_qu(:, :, i));

            % Calculate EC results using linear equations
            A = [
                1, 0, 0, 0, 0, 1, 0, 0, 0, 0;
                0, 1, 0, 0, 0, 0, 1, 0, 0, 0;
                0, 0, 1, 0, 0, 0, 0, 1, 0, 0;
                0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
                0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
            ];

            y = [
                diag(ExxT);
                ExxT(1, 2);
                ExxT(1, 3) * ExxT(1, 4) / ExxT(3, 4);
                ExxT(2, 3) * ExxT(2, 4) / ExxT(3, 4);
                ExxT(1, 3) * ExxT(3, 4) / ExxT(1, 4);
                ExxT(2, 3) * ExxT(3, 4) / ExxT(2, 4);
                ExxT(1, 4) * ExxT(3, 4) / ExxT(1, 3);
                ExxT(2, 4) * ExxT(3, 4) / ExxT(2, 3);
                ExxT(1, 3) * ExxT(2, 4) / ExxT(3, 4);
                ExxT(1, 4) * ExxT(2, 3) / ExxT(3, 4);
            ];
            x = ((A' * A) \ A' * y);

            % Assemble error variance matrix
            EeeT = diag(x(6:9));
            EeeT(1, 2) = x(end);
            EeeT(2, 1) = x(end);

            % Calculate Signal-to-Noise Ratio (SNR) and Fractional Mean Squared Error (fMSE)
            SNR = x(1:4) ./ x(6:9);
            rho2 = SNR ./ (1 + SNR);
            fMSE = 1 - rho2;

            % Solve weight
%             eta = ones(size(EeeT, 2), 1);
%             re_weight = (eta' * (EeeT \ eta)) \ (EeeT \ eta)';
            var = diag(EeeT);
            covar = EeeT(1, 2);
            
            z = (var(1) + var(2) - 2 * covar) / (var(1) * var(2) - covar^2) + 1 / var(3) + 1/var(4);
            re_weight(1) = (var(2) - covar) / ((var(1) * var(2) - covar^2) * z);
            re_weight(2) = (var(1) - covar) / ((var(1) * var(2) - covar^2) * z);
            re_weight(3) = 1 / (var(3) * z);
            re_weight(4) = 1 / (var(4) * z);
            % Calculate weighted average result
            weighted_result = EC_merge_ini(qu, res_qu(:, :, i), re_weight, invalid_rows);

            % Store EC results in struct array
            results(comb_idx).EeeT{i} = EeeT;
            results(comb_idx).SNR{i} = SNR;
            results(comb_idx).rho2{i} = rho2;
            results(comb_idx).fMSE{i} = fMSE;
            results(comb_idx).re_weight{i} = re_weight;
            results(comb_idx).weighted_result{i} = weighted_result;
        end
    end
else
    error('Function is only valid for quadruple situation.');
end
end

function mergeResult = EC_merge_ini(qu, res_qu, re_weight, invalid_rows)
    mergeResult = nan(size(qu, 1), 1);
    
    % Calculate average values for invalid rows
    averageValues = nanmean(qu(invalid_rows, :), 2);
    
    % Calculate weighted average results for valid rows
    weightedResults = sum(res_qu .* re_weight, 2);
    
    % Fill mergeResult with average values for invalid rows
    mergeResult(invalid_rows) = averageValues;
    
    % Fill mergeResult with weighted results for valid rows
    mergeResult(~invalid_rows) = weightedResults;
end
