clear;clc;
addpath('Z:\LCM\Collocation\Functions');
NaN_name = fullfile('Z:\LCM\ELI\data\ELI\Merma','Merma.self.1996.nc');
NaN_info = ncread(NaN_name,'EIL_root');
% b = NaN_info(:,721:end);
% d = NaN_info(:,1:720); 
% NaN_info(:,1:720) = b;
% NaN_info(:,721:end) = d;
[rr,cc] = find(~isnan(NaN_info));
%% ERA5L+GLEAM 1980.01.01-1999.12.31 dual 4-IVD Et/Tveg/SM
Y = (1980:1:1999);
combine_nsma = nan(600,1440,240,2);
combine_ssma = nan(600,1440,240,2);
combine_eta = nan(600,1440,240,2);
combine_tvega = nan(600,1440,240,2);
merge_swa = nan(600,1440,240);
for y_idx = 1:length(Y)
    for mon = 1:12
        mon_name = 100 + mon; mon_name = num2str(mon_name);
        mon_name = mon_name(2:3);
        time = strcat(num2str(Y(y_idx)),mon_name);
        ELa = strcat('ERA5L025.',time,'.nc');
        ELa = fullfile('Z:\LCM\ELI\data\Monthly_anomaly\ERA5Lto025',ELa);
        G38a = strcat('G38a.',time,'.nc');
        G38a = fullfile('Z:\LCM\ELI\data\Monthly_anomaly\GLEAM38a',G38a);
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
        % put inside
        combine_nsma(:,:,mon+(y_idx-1)*12,1) = nsma_ELa;
        combine_ssma(:,:,mon+(y_idx-1)*12,1) = ssma_ELa;
        combine_tvega(:,:,mon+(y_idx-1)*12,1) = tvega_ELa;
        combine_eta(:,:,mon+(y_idx-1)*12,1) = eta_ELa;
        combine_nsma(:,:,mon+(y_idx-1)*12,2) = nsma_G38a;
        combine_ssma(:,:,mon+(y_idx-1)*12,2) = ssma_G38a;
        combine_tvega(:,:,mon+(y_idx-1)*12,2) = tvega_G38a;
        combine_eta(:,:,mon+(y_idx-1)*12,2) = eta_G38a;
        merge_swa(:,:,mon+(y_idx-1)*12) = swa_ELa;
        disp(['Put inside ',time]);
    end
end
% IVD calculation
sige2_nsma = nan(600,1440,2); rho2_nsma = nan(600,1440,2); 
weight_nsma = nan(600,1440,2); merge_nsma = nan(600,1440,240);
sige2_ssma = nan(600,1440,2); rho2_ssma = nan(600,1440,2); 
weight_ssma = nan(600,1440,2); merge_ssma = nan(600,1440,240);
sige2_tvega = nan(600,1440,2); rho2_tvega = nan(600,1440,2); 
weight_tvega = nan(600,1440,2); merge_tvega = nan(600,1440,240);
sige2_eta = nan(600,1440,2); rho2_eta = nan(600,1440,2); 
weight_eta = nan(600,1440,2); merge_eta = nan(600,1440,240);
for i = 1:length(rr)
    row = rr(i); col = cc(i);
    dual_nsma = combine_nsma(row,col,:,:); dual_nsma = reshape(dual_nsma,[240,2]); 
    dual_nsma(isnan(dual_nsma)) = mean(mean(dual_nsma,'omitnan'),'omitnan');
    dual_nsma = dual_nsma(all(~isnan(dual_nsma),2),:);
    dual_ssma = combine_ssma(row,col,:,:); dual_ssma = reshape(dual_ssma,[240,2]); 
    dual_ssma(isnan(dual_ssma)) = mean(mean(dual_ssma,'omitnan'),'omitnan');
    dual_ssma = dual_ssma(all(~isnan(dual_ssma),2),:);
    dual_tvega = combine_tvega(row,col,:,:); dual_tvega = reshape(dual_tvega,[240,2]); 
    dual_tvega(isnan(dual_tvega)) = mean(mean(dual_tvega,'omitnan'),'omitnan');
    dual_tvega = dual_tvega(all(~isnan(dual_tvega),2),:);
    dual_eta = combine_eta(row,col,:,:); dual_eta = reshape(dual_eta,[240,2]); 
    dual_eta(isnan(dual_eta)) = mean(mean(dual_eta,'omitnan'),'omitnan');
    dual_eta = dual_eta(all(~isnan(dual_eta),2),:);
    % use IVD method
    % nsma
    [EeeT,rho2,u] = IVD(dual_nsma);
    sige2_nsma(row,col,1) = EeeT(1,1); sige2_nsma(row,col,2) = EeeT(2,2);
    rho2_nsma(row,col,1) = rho2(1); rho2_nsma(row,col,2) = rho2(2);
    weight_nsma(row,col,1) = u(1); weight_nsma(row,col,2) = u(2);
    mean_nsma = mean(dual_nsma,2,'omitnan');
    u_nsma = dual_nsma(:,1)*u(1) + dual_nsma(:,2)*u(2);
    idx = find(isnan(u_nsma));
    u_nsma(idx) = mean_nsma(idx);
    if ~isempty(u_nsma)
        merge_nsma(row,col,:) = u_nsma;
    else
        merge_nsma(row,col,:) = nan(240,1);
    end
    % ssma
    [EeeT,rho2,u] = IVD(dual_ssma);
    sige2_ssma(row,col,1) = EeeT(1,1); sige2_ssma(row,col,2) = EeeT(2,2);
    rho2_ssma(row,col,1) = rho2(1); rho2_ssma(row,col,2) = rho2(2);
    weight_ssma(row,col,1) = u(1); weight_nsma(row,col,2) = u(2);
    mean_ssma = mean(dual_ssma,2,'omitnan');
    u_ssma = dual_ssma(:,1)*u(1) + dual_ssma(:,2)*u(2);
    idx = find(isnan(u_ssma));
    u_ssma(idx) = mean_ssma(idx);
    if ~isempty(u_ssma)
        merge_ssma(row,col,:) = u_ssma;
    else
        merge_ssma(row,col,:) = nan(240,1);
    end
    % tvega
    [EeeT,rho2,u] = IVD(dual_tvega);
    sige2_tvega(row,col,1) = EeeT(1,1); sige2_tvega(row,col,2) = EeeT(2,2);
    rho2_tvega(row,col,1) = rho2(1); rho2_tvega(row,col,2) = rho2(2);
    weight_tvega(row,col,1) = u(1); weight_tvega(row,col,2) = u(2);
    mean_tvega = mean(dual_tvega,2,'omitnan');
    u_tvega = dual_tvega(:,1)*u(1) + dual_tvega(:,2)*u(2);
    idx = find(isnan(u_tvega));
    u_tvega(idx) = mean_tvega(idx);
    if ~isempty(u_tvega)
        merge_tvega(row,col,:) = u_tvega;
    else
        merge_tvega(row,col,:) = nan(240,1);
    end
    % eta
    [EeeT,rho2,u] = IVD(dual_eta);
    sige2_eta(row,col,1) = EeeT(1,1); sige2_eta(row,col,2) = EeeT(2,2);
    rho2_eta(row,col,1) = rho2(1); rho2_eta(row,col,2) = rho2(2);
    weight_eta(row,col,1) = u(1); weight_eta(row,col,2) = u(2);
    mean_eta = mean(dual_eta,2,'omitnan');
    u_eta = dual_eta(:,1)*u(1) + dual_eta(:,2)*u(2);
    idx = find(isnan(u_eta));
    u_eta(idx) = mean_eta(idx);
    if ~isempty(u_eta)
        merge_eta(row,col,:) = u_eta;
    else
        merge_eta(row,col,:) = nan(240,1);
    end
    disp(['EL+G38a - ',num2str(i),'/',num2str(length(rr))]);
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
                merge_swa(:,:,mon+(y_idx-1)*12,1),...,
                merge_eta(:,:,mon+(y_idx-1)*12,1));
            disp(['Process EL+G38a - ',time]);
        else
            disp(['DoneEL+G38a - ',time]);
        end
    end
end
nc_error_out('IVD_EL_nsma.nc',sige2_nsma(:,:,1),rho2_nsma(:,:,1),weight_nsma(:,:,1));
nc_error_out('IVD_G38a_nsma.nc',sige2_nsma(:,:,2),rho2_nsma(:,:,2),weight_nsma(:,:,2));
nc_error_out('IVD_EL_ssma.nc',sige2_ssma(:,:,1),rho2_ssma(:,:,1),weight_ssma(:,:,1));
nc_error_out('IVD_G38a_ssma.nc',sige2_ssma(:,:,2),rho2_ssma(:,:,2),weight_ssma(:,:,2));
nc_error_out('IVD_EL_tvega.nc',sige2_tvega(:,:,1),rho2_tvega(:,:,1),weight_tvega(:,:,1));
nc_error_out('IVD_G38a_tvega.nc',sige2_tvega(:,:,2),rho2_tvega(:,:,2),weight_tvega(:,:,2));
nc_error_out('IVD_EL_eta.nc',sige2_eta(:,:,1),rho2_eta(:,:,1),weight_eta(:,:,1));
nc_error_out('IVD_G38a_eta.nc',sige2_eta(:,:,2),rho2_eta(:,:,2),weight_eta(:,:,2));
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