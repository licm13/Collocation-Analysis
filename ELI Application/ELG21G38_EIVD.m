clear;clc;
addpath('Z:\LCM\Collocation\Functions');
NaN_name = fullfile('Z:\LCM\ELI\data\ELI\Merma','Merma.self.1996.nc');
NaN_info = ncread(NaN_name,'EIL_root');
% b = NaN_info(:,721:end);
% d = NaN_info(:,1:720); 
% NaN_info(:,1:720) = b;
% NaN_info(:,721:end) = d;
[rr,cc] = find(~isnan(NaN_info));
%% ERA5L+GLEAM+GLDAS21 2000.01.01-2022.12.31 ET/T/SM-EIVD SW-IVD
Y = (2000:1:2022);
combine_nsma = nan(600,1440,276,3);
combine_ssma = nan(600,1440,276,3);
combine_eta = nan(600,1440,276,3);
combine_tvega = nan(600,1440,276,3);
combine_swa = nan(600,1440,276,2); % only ERA5L G21
for y_idx = 1:length(Y)
    for mon = 1:12
        mon_name = 100 + mon; mon_name = num2str(mon_name);
        mon_name = mon_name(2:3);
        time = strcat(num2str(Y(y_idx)),mon_name);
        ELa = strcat('ERA5L025.',time,'.nc');
        ELa = fullfile('Z:\LCM\ELI\data\Monthly_anomaly\ERA5Lto025',ELa);
        G38a = strcat('G38a.',time,'.nc');
        G38a = fullfile('Z:\LCM\ELI\data\Monthly_anomaly\GLEAM38a',G38a);
        G21a = strcat('G21.',time,'.nc');
        G21a = fullfile('Z:\LCM\ELI\data\Monthly_anomaly\GLDAS21',G21a);
        % ERA5L
        nsma_ELa = ncread(ELa,'nsma');
        ssma_ELa = ncread(ELa,'ssma');
        tvega_ELa = ncread(ELa,'tvega');
        eta_ELa = ncread(ELa,'eta');
        swa_ELa = ncread(ELa,'swa');
        % G38a
        nsma_G38a = ncread(G38a,'nsma');
        ssma_G38a = ncread(G38a,'ssma');
        tvega_G38a = ncread(G38a,'tvega');
        eta_G38a = ncread(G38a,'eta');
        % G21a
        nsma_G21a = ncread(G21a,'nsma');
        ssma_G21a = ncread(G21a,'ssma');
        tvega_G21a = ncread(G21a,'tvega');
        eta_G21a = ncread(G21a,'eta');
        swa_G21a = ncread(G21a,'swa');
        % put inside
        combine_nsma(:,:,mon+(y_idx-1)*12,1) = nsma_ELa;
        combine_ssma(:,:,mon+(y_idx-1)*12,1) = ssma_ELa;
        combine_tvega(:,:,mon+(y_idx-1)*12,1) = tvega_ELa;
        combine_eta(:,:,mon+(y_idx-1)*12,1) = eta_ELa;
        
        combine_nsma(:,:,mon+(y_idx-1)*12,2) = nsma_G38a;
        combine_ssma(:,:,mon+(y_idx-1)*12,2) = ssma_G38a;
        combine_tvega(:,:,mon+(y_idx-1)*12,2) = tvega_G38a;
        combine_eta(:,:,mon+(y_idx-1)*12,2) = eta_G38a;
        
        combine_nsma(:,:,mon+(y_idx-1)*12,3) = nsma_G21a;
        combine_ssma(:,:,mon+(y_idx-1)*12,3) = ssma_G21a;
        combine_tvega(:,:,mon+(y_idx-1)*12,3) = tvega_G21a;
        combine_eta(:,:,mon+(y_idx-1)*12,3) = eta_G21a;
        % swa only EL G21a
        combine_swa(:,:,mon+(y_idx-1)*12,1) = swa_ELa;
        combine_swa(:,:,mon+(y_idx-1)*12,2) = swa_G21a;
        disp(['Put inside ',time]);
    end
end
% EIVD calculation
sige2_nsma = nan(600,1440,3); rho2_nsma = nan(600,1440,3); 
weight_nsma = nan(600,1440,3); merge_nsma = nan(600,1440,276);
ecc_nsma = nan(600,1440,3);
sige2_ssma = nan(600,1440,3); rho2_ssma = nan(600,1440,3); 
weight_ssma = nan(600,1440,3); merge_ssma = nan(600,1440,276);
ecc_ssma = nan(600,1440,3);
sige2_tvega = nan(600,1440,3); rho2_tvega = nan(600,1440,3); 
weight_tvega = nan(600,1440,3); merge_tvega = nan(600,1440,276);
ecc_tvega = nan(600,1440,3);
sige2_eta = nan(600,1440,3); rho2_eta = nan(600,1440,3); 
weight_eta = nan(600,1440,3); merge_eta = nan(600,1440,276);
ecc_eta = nan(600,1440,3);
sige2_swa = nan(600,1440,2); rho2_swa = nan(600,1440,2); 
weight_swa = nan(600,1440,2); merge_swa = nan(600,1440,276);
for i = 1:length(rr)
    %% EIVD: T/ET/SM
    row = rr(i); col = cc(i);
    tri_nsma = combine_nsma(row,col,:,:); tri_nsma = reshape(tri_nsma,[276,3]); 
    tri_nsma(isnan(tri_nsma)) = mean(mean(tri_nsma,'omitnan'),'omitnan');
    tri_nsma = tri_nsma(all(~isnan(tri_nsma),2),:);
    if isempty(tri_nsma)
        tri_nsma = rand(276,3);
    end
    tri_ssma = combine_ssma(row,col,:,:); tri_ssma = reshape(tri_ssma,[276,3]); 
    tri_ssma(isnan(tri_ssma)) = mean(mean(tri_ssma,'omitnan'),'omitnan');
    tri_ssma = tri_ssma(all(~isnan(tri_ssma),2),:);
    if isempty(tri_ssma)
        tri_ssma = rand(276,3);
    end
    tri_tvega = combine_tvega(row,col,:,:); tri_tvega = reshape(tri_tvega,[276,3]); 
    tri_tvega(isnan(tri_tvega)) = mean(mean(tri_tvega,'omitnan'),'omitnan');
    tri_tvega = tri_tvega(all(~isnan(tri_tvega),2),:);
    if isempty(tri_tvega)
        tri_tvega = rand(276,3);
    end
    tri_eta = combine_eta(row,col,:,:); tri_eta = reshape(tri_eta,[276,3]); 
    tri_eta(isnan(tri_eta)) = mean(mean(tri_eta,'omitnan'),'omitnan');
    tri_eta = tri_eta(all(~isnan(tri_eta),2),:);
    if isempty(tri_eta)
        tri_eta = rand(276,3);
    end
    % nsma
    r = EIVD_alltogether(tri_nsma); r = r(1);
    sige2_nsma(row,col,:) = diag(r.EeeT{1,1});
    rho2_nsma(row,col,:) = r.rho2{1,1};
    weight_nsma(row,col,:) = r.re_weight{1,1}';
    merge_nsma(row,col,:) = r.weighted_result{1,1};
    ecc_nsma(row,col,1) = r.EeeT{1,1}(2,3);
    ecc_nsma(row,col,2) = r.EeeT{1,2}(2,3);
    ecc_nsma(row,col,2) = r.EeeT{1,3}(2,3);
    % ssma
    r = EIVD_alltogether(tri_ssma); r = r(1);
    sige2_ssma(row,col,:) = diag(r.EeeT{1,1});
    rho2_ssma(row,col,:) = r.rho2{1,1};
    weight_ssma(row,col,:) = r.re_weight{1,1}';
    merge_ssma(row,col,:) = r.weighted_result{1,1};
    ecc_ssma(row,col,1) = r.EeeT{1,1}(2,3);
    ecc_ssma(row,col,2) = r.EeeT{1,2}(2,3);
    ecc_ssma(row,col,2) = r.EeeT{1,3}(2,3);
    % tvega
    r = EIVD_alltogether(tri_tvega); r = r(1);
    sige2_tvega(row,col,:) = diag(r.EeeT{1,1});
    rho2_tvega(row,col,:) = r.rho2{1,1};
    weight_tvega(row,col,:) = r.re_weight{1,1}';
    merge_tvega(row,col,:) = r.weighted_result{1,1};
    ecc_tvega(row,col,1) = r.EeeT{1,1}(2,3);
    ecc_tvega(row,col,2) = r.EeeT{1,2}(2,3);
    ecc_tvega(row,col,2) = r.EeeT{1,3}(2,3);
    % eta
    r = EIVD_alltogether(tri_eta); r = r(1);
    sige2_eta(row,col,:) = diag(r.EeeT{1,1});
    rho2_eta(row,col,:) = r.rho2{1,1};
    weight_eta(row,col,:) = r.re_weight{1,1}';
    merge_eta(row,col,:) = r.weighted_result{1,1};
    ecc_eta(row,col,1) = r.EeeT{1,1}(2,3);
    ecc_eta(row,col,2) = r.EeeT{1,2}(2,3);
    ecc_eta(row,col,2) = r.EeeT{1,3}(2,3);
    disp(['EL+G38a+G21a - ',num2str(i),'/',num2str(length(rr))]);
    %% IVD: SWin only EL G21a
    dual_swa = combine_swa(row,col,:,:); dual_swa = reshape(dual_swa,[276,2]); 
    dual_swa(isnan(dual_swa)) = mean(mean(dual_swa,'omitnan'),'omitnan');
    dual_swa = dual_swa(all(~isnan(dual_swa),2),:);
    % swa
    [EeeT,rho2,u] = IVD(dual_swa);
    sige2_swa(row,col,1) = EeeT(1,1); sige2_swa(row,col,2) = EeeT(2,2);
    rho2_swa(row,col,1) = rho2(1); rho2_swa(row,col,2) = rho2(2);
    weight_swa(row,col,1) = u(1); weight_swa(row,col,2) = u(2);
    mean_swa = mean(dual_swa,2,'omitnan');
    u_swa = dual_swa(:,1)*u(1) + dual_swa(:,2)*u(2);
    idx = find(isnan(u_swa));
    u_swa(idx) = mean_swa(idx);
    if ~isempty(u_swa)
        merge_swa(row,col,:) = u_swa;
    else
        merge_swa(row,col,:) = nan(276,1);
    end
    disp(['EL+G21a - ',num2str(i),'/',num2str(length(rr))]);
end
% write-out
for y_idx = 1:length(Y)
    for mon = 1:12
        mon_name = 100 + mon; mon_name = num2str(mon_name);
        mon_name = mon_name(2:3);
        time = strcat(num2str(Y(y_idx)),mon_name);
        outfile = strcat('Collocation_Merma.',time,'.nc');
        outfile = fullfile('Z:\LCM\ELI\data\Monthly_anomaly\Collocation_Merma',outfile);
        if exist(outfile,'file') == 0
            nc_out(outfile,merge_nsma(:,:,mon+(y_idx-1)*12,1),...,
                merge_ssma(:,:,mon+(y_idx-1)*12,1),...,
                merge_tvega(:,:,mon+(y_idx-1)*12,1),...,
                combine_swa(:,:,mon+(y_idx-1)*12,1),...,
                merge_eta(:,:,mon+(y_idx-1)*12,1));
            disp(['Process EL+G38a+G21a - ',time]);
        else
            disp(['DoneEL+G38a+G21a - ',time]);
        end
    end
end
nc_error_out('EIVD_EL_nsma.nc',sige2_nsma(:,:,1),rho2_nsma(:,:,1),weight_nsma(:,:,1));
nc_error_out('EIVD_G38a_nsma.nc',sige2_nsma(:,:,2),rho2_nsma(:,:,2),weight_nsma(:,:,2));
nc_error_out('EIVD_G21a_nsma.nc',sige2_nsma(:,:,3),rho2_nsma(:,:,3),weight_nsma(:,:,3));
nc_ecc_out('ECC_EL_G38a_nsma.nc',ecc_nsma(:,:,1));
nc_ecc_out('ECC_EL_G21a_nsma.nc',ecc_nsma(:,:,2));
nc_ecc_out('ECC_G38a_G21a_nsma.nc',ecc_nsma(:,:,3));
nc_error_out('EIVD_EL_ssma.nc',sige2_ssma(:,:,1),rho2_ssma(:,:,1),weight_ssma(:,:,1));
nc_error_out('EIVD_G38a_ssma.nc',sige2_ssma(:,:,2),rho2_ssma(:,:,2),weight_ssma(:,:,2));
nc_error_out('EIVD_G21a_ssma.nc',sige2_ssma(:,:,3),rho2_ssma(:,:,3),weight_ssma(:,:,3));
nc_ecc_out('ECC_EL_G38a_ssma.nc',ecc_ssma(:,:,1));
nc_ecc_out('ECC_EL_G21a_ssma.nc',ecc_ssma(:,:,2));
nc_ecc_out('ECC_G38a_G21a_ssma.nc',ecc_ssma(:,:,3));
nc_error_out('EIVD_EL_tvega.nc',sige2_tvega(:,:,1),rho2_tvega(:,:,1),weight_tvega(:,:,1));
nc_error_out('EIVD_G38a_tvega.nc',sige2_tvega(:,:,2),rho2_tvega(:,:,2),weight_tvega(:,:,2));
nc_error_out('EIVD_G21a_tvega.nc',sige2_tvega(:,:,3),rho2_tvega(:,:,3),weight_tvega(:,:,3));
nc_ecc_out('ECC_EL_G38a_tvega.nc',ecc_tvega(:,:,1));
nc_ecc_out('ECC_EL_G21a_tvega.nc',ecc_tvega(:,:,2));
nc_ecc_out('ECC_G38a_G21a_tvega.nc',ecc_tvega(:,:,3));
nc_error_out('EIVD_EL_eta.nc',sige2_eta(:,:,1),rho2_eta(:,:,1),weight_eta(:,:,1));
nc_error_out('EIVD_G38a_eta.nc',sige2_eta(:,:,2),rho2_eta(:,:,2),weight_eta(:,:,2));
nc_error_out('EIVD_G21a_eta.nc',sige2_eta(:,:,3),rho2_eta(:,:,3),weight_eta(:,:,3));
nc_ecc_out('ECC_EL_G38a_eta.nc',ecc_eta(:,:,1));
nc_ecc_out('ECC_EL_G21a_eta.nc',ecc_eta(:,:,2));
nc_ecc_out('ECC_G38a_G21a_eta.nc',ecc_eta(:,:,3));
nc_error_out('IVD_EL_swa.nc',sige2_swa(:,:,1),rho2_swa(:,:,1),weight_swa(:,:,1));
nc_error_out('IVD_G21a_swa.nc',sige2_swa(:,:,2),rho2_swa(:,:,2),weight_swa(:,:,2));
%% Funcion region~~
function nc_out(outfile,nsma,ssma,tvega,swa,eta)
    nsma = deal_complex(nsma);
    ssma = deal_complex(ssma);
    tvega = deal_complex(tvega);
    swa = deal_complex(swa);
    eta = deal_complex(eta);
    lat = (89.75:-0.25:-60);
    lon = (179.75:-0.25:-180);
    row = 600; col = 1440;        
    % lon attribute
    nccreate(outfile,'lon','datatype','double','Dimensions',{'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'lon','units','degrees_east');
    ncwriteatt(outfile,'lon','long_name','longitude');
    ncwrite(outfile,'lon',lon);
    % lat attribute
    nccreate(outfile,'lat','datatype','double','Dimensions',{'lat',row},'DeflateLevel',5);
    ncwriteatt(outfile,'lat','units','degrees_north');
    ncwriteatt(outfile,'lat','long_name','latitude');
    ncwrite(outfile,'lat',lat);
    % nsma attribute
    nccreate(outfile,'nsma','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'nsma','unit','mm3 mm-3');
    ncwriteatt(outfile,'nsma','long_name','monthly near surface SM anomaly (month_value - annual_mean) (0-10cm)');
    ncwriteatt(outfile,'nsma','missing_value',nan);
    ncwriteatt(outfile,'nsma','FillValue',nan);
    ncwriteatt(outfile,'nsma','scale_factor',1);
    ncwriteatt(outfile,'nsma','add_offset',0);
    ncwrite(outfile,'nsma',nsma);
    % ssma attribute
    nccreate(outfile,'ssma','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'ssma','unit','mm3 mm-3');
    ncwriteatt(outfile,'ssma','long_name','monthly sub surface SM anomaly (month_value - annual_mean) (mean 10-100cm)');
    ncwriteatt(outfile,'ssma','missing_value',nan);
    ncwriteatt(outfile,'ssma','FillValue',nan);
    ncwriteatt(outfile,'ssma','scale_factor',1);
    ncwriteatt(outfile,'ssma','add_offset',0);
    ncwrite(outfile,'ssma',ssma);
    % tvega attribute
    nccreate(outfile,'tvega','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'tvega','unit','mm mon-1');
    ncwriteatt(outfile,'tvega','long_name','monthly transpiration anomaly (month_value - annual_mean)');
    ncwriteatt(outfile,'tvega','missing_value',nan);
    ncwriteatt(outfile,'tvega','FillValue',nan);
    ncwriteatt(outfile,'tvega','scale_factor',1);
    ncwriteatt(outfile,'tvega','add_offset',0);
    ncwrite(outfile,'tvega',tvega);
    % swa attribute
    nccreate(outfile,'swa','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'swa','unit','J m-2');
    ncwriteatt(outfile,'swa','long_name','monthly Downward short-wave radiation flux anomaly (month_value - annual_mean)');
    ncwriteatt(outfile,'swa','missing_value',nan);
    ncwriteatt(outfile,'swa','FillValue',nan);
    ncwriteatt(outfile,'swa','scale_factor',1);
    ncwriteatt(outfile,'swa','add_offset',0);
    ncwrite(outfile,'swa',swa);
    % eta attribute
    nccreate(outfile,'eta','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'eta','unit','mm mon-1');
    ncwriteatt(outfile,'eta','long_name','monthly total ET anomaly (month_value - annual_mean)');
    ncwriteatt(outfile,'eta','missing_value',nan);
    ncwriteatt(outfile,'eta','FillValue',nan);
    ncwriteatt(outfile,'eta','scale_factor',1);
    ncwriteatt(outfile,'eta','add_offset',0);
    ncwrite(outfile,'eta',eta);
end
function nc_error_out(outfile,sige2,rho2,u)
    sige2 = deal_complex(sige2);
    rho2 = deal_complex(rho2);
    u = deal_complex(u);
    lat = (89.75:-0.25:-60);
    lon = (179.75:-0.25:-180);
    row = 600; col = 1440;        
    % lon attribute
    nccreate(outfile,'lon','datatype','double','Dimensions',{'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'lon','units','degrees_east');
    ncwriteatt(outfile,'lon','long_name','longitude');
    ncwrite(outfile,'lon',lon);
    % lat attribute
    nccreate(outfile,'lat','datatype','double','Dimensions',{'lat',row},'DeflateLevel',5);
    ncwriteatt(outfile,'lat','units','degrees_north');
    ncwriteatt(outfile,'lat','long_name','latitude');
    ncwrite(outfile,'lat',lat);
    % sige2 attribute
    nccreate(outfile,'sige2','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'sige2','missing_value',nan);
    ncwriteatt(outfile,'sige2','FillValue',nan);
    ncwriteatt(outfile,'sige2','scale_factor',1);
    ncwriteatt(outfile,'sige2','add_offset',0);
    ncwrite(outfile,'sige2',sige2);
    % rho2 attribute
    nccreate(outfile,'rho2','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'rho2','missing_value',nan);
    ncwriteatt(outfile,'rho2','FillValue',nan);
    ncwriteatt(outfile,'rho2','scale_factor',1);
    ncwriteatt(outfile,'rho2','add_offset',0);
    ncwrite(outfile,'rho2',rho2);
    % u attribute
    nccreate(outfile,'u','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'u','missing_value',nan);
    ncwriteatt(outfile,'u','FillValue',nan);
    ncwriteatt(outfile,'u','scale_factor',1);
    ncwriteatt(outfile,'u','add_offset',0);
    ncwrite(outfile,'u',u);
end
function nc_ecc_out(outfile,ecc)
    ecc = deal_complex(ecc);
    lat = (89.75:-0.25:-60);
    lon = (179.75:-0.25:-180);
    row = 600; col = 1440;        
    % lon attribute
    nccreate(outfile,'lon','datatype','double','Dimensions',{'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'lon','units','degrees_east');
    ncwriteatt(outfile,'lon','long_name','longitude');
    ncwrite(outfile,'lon',lon);
    % lat attribute
    nccreate(outfile,'lat','datatype','double','Dimensions',{'lat',row},'DeflateLevel',5);
    ncwriteatt(outfile,'lat','units','degrees_north');
    ncwriteatt(outfile,'lat','long_name','latitude');
    ncwrite(outfile,'lat',lat);
    % ecc attribute
    nccreate(outfile,'ecc','datatype','double','Dimensions',{'lat',row,'lon',col},'DeflateLevel',5);
    ncwriteatt(outfile,'ecc','missing_value',nan);
    ncwriteatt(outfile,'ecc','FillValue',nan);
    ncwriteatt(outfile,'ecc','scale_factor',1);
    ncwriteatt(outfile,'ecc','add_offset',0);
    ncwrite(outfile,'ecc',ecc);
end
function [data] = deal_complex(data)
    ce = imag(data)~=0;
    data(ce) = nan;
end