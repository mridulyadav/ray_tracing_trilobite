% clearvars -except PathName FileName;
% clc;
% [FileName,PathName] = uigetfile('*.*','Select the Input file');


% Mass = xlsread(strcat(PathName,FileName),'Sheet1','B2');
% Temperature = xlsread(strcat(PathName,FileName),'Sheet1','B3');
clear;

[FileName2,PathName2] = uigetfile('*.*','Select Input File');
[status,sheets] = xlsfinfo(strcat(PathName2,FileName2));

[s,v] = listdlg('PromptString','Select a Sheet Name:',...
                'SelectionMode','single',...
                'ListString',sheets)

% Sheet_full = {'150_deg_900um_5000_s_s_A.pn'}; %Sheet Name
% M_full = [0.23];
% WR_full = zeros(1,53);
% size_M = size(M_full);

%for i = 1:size_M(2)
Sheet_temp = sheets(s);
Sheet = Sheet_temp{1};

prompt = {'Mass (mg) : ','Temperature (C) : '};
dlg_title = 'Mass and Temperature';
num_lines = [1 50];
% Substitute reference size in micron and length of filename; default = {'Reference size in microns', 'Length of filename (max = 31)'}
default = {'15','150'}; 
answer = inputdlg(prompt,dlg_title,num_lines,default);

input_data = zeros(2,1);

for k = 1:2
    input_data(k,1) = str2num(answer{k,:});
end
Mass = input_data(1,1);% in microns
Temperature = input_data(2,1);
    
% PathName2 = 'D:\Mridul''s\Documents\M.Tech\Aspheric Lens\Drawing lenses\5000stepsec_drawing_lens_900_um_ndldia\Cropped\';
% FileName2 = 'Results_5000stepsec_drawing_lens_900_um_ndldia_WithSymmetry.xlsx';

X1 = xlsread(strcat(PathName2,FileName2),Sheet,'C:E');
X = [X1(:,1) X1(:,3)];
R = xlsread(strcat(PathName2,FileName2),Sheet,'B5');
ConicConst_k = xlsread(strcat(PathName2,FileName2),Sheet,'B6');
% X = [X1(:,1) X1(:,3)];
% R = Q(1);
% k = Q(2);
% ConicConst_k = k;
% Mass = M;
% Temperature = 200;

PathName = PathName2;
FileName = strcat(Sheet,'.xlsx');

t = 0;
script_RayTracing_Infinity_FFL
xlswrite(strcat(PathName,FileName),{'Forward Focal Length_No Glass Plate'},'RayTracing Results','A1')
A = {'Temperature(deg C)','Mass(mg)','R(micron)','k','Glass Plate Thickness_t(micron)','Paraxial Focal Length(micron)', ...
    'Surface of Least Confusion(micron)','Spot Size_dia(micron)','Longitudinal Spherical Aberration_max(micron)', ...
    'Longitudinal Spherical Aberration Coefficient','Transverse Spherical Aberration_max(micron)', ...
    'Transverse Spherical Aberration Coefficient','Longitudinal Chromatic Aberration(micron)';Temperature,Mass,R,ConicConst_k,t,Paraxial_focal_length, ...
    z_lcs,beam_t_min,LSA_max,a_lsa,TSA_max,a_tsa,Longitudinal_Chromatic_Aberration};
xlswrite(strcat(PathName,FileName),A,'RayTracing Results','A2')

WR = [Temperature Mass R ConicConst_k t Paraxial_focal_length ...
    z_lcs beam_t_min LSA_max a_lsa TSA_max a_tsa Longitudinal_Chromatic_Aberration];

clearvars -except FileName PathName X1 WR WR_full M_full Sheet_full Q M X R ConicConst_k Mass Temperature 
t = 1000;
script_RayTracing_Infinity_FFL
xlswrite(strcat(PathName,FileName),{'Forward Focal Length_With Glass Plate'},'RayTracing Results','A4')
A = {'Temperature(deg C)','Mass(mg)','R(micron)','k','Glass Plate Thickness_t(micron)','Paraxial Focal Length(micron)', ...
    'Surface of Least Confusion(micron)','Spot Size_dia(micron)','Longitudinal Spherical Aberration_max(micron)', ...
    'Longitudinal Spherical Aberration Coefficient','Transverse Spherical Aberration_max(micron)', ...
    'Transverse Spherical Aberration Coefficient','Longitudinal Chromatic Aberration(micron)';Temperature,Mass,R,ConicConst_k,t,Paraxial_focal_length, ...
    z_lcs,beam_t_min,LSA_max,a_lsa,TSA_max,a_tsa,Longitudinal_Chromatic_Aberration};
xlswrite(strcat(PathName,FileName),A,'RayTracing Results','N2')

WR = [WR [Temperature Mass R ConicConst_k t Paraxial_focal_length ...
    z_lcs beam_t_min LSA_max a_lsa TSA_max a_tsa Longitudinal_Chromatic_Aberration]];

clearvars -except WR WR_full M_full Sheet_full FileName PathName X1 Q M X R ConicConst_k Mass Temperature
t = 0;
script_RayTracing_Infinity_BFL
xlswrite(strcat(PathName,FileName),{'Backward Focal Length_No Glass Plate'},'RayTracing Results','A8')
A = {'Temperature(deg C)','Mass(mg)','R(micron)','k','Glass Plate Thickness_t(micron)','Paraxial Focal Length(micron)', ...
    'Surface of Least Confusion(micron)','Spot Size_dia(micron)','Longitudinal Spherical Aberration_max(micron)', ...
    'Longitudinal Spherical Aberration Coefficient','Transverse Spherical Aberration_max(micron)', ...
    'Transverse Spherical Aberration Coefficient','Longitudinal Chromatic Aberration(micron)';Temperature,Mass,R,ConicConst_k,t,Paraxial_focal_length, ...
    z_lcs,beam_t_min,LSA_max,a_lsa,TSA_max,a_tsa,Longitudinal_Chromatic_Aberration};
xlswrite(strcat(PathName,FileName),A,'RayTracing Results','AA2')

WR = [WR [Temperature Mass R ConicConst_k t Paraxial_focal_length ...
    z_lcs beam_t_min LSA_max a_lsa TSA_max a_tsa Longitudinal_Chromatic_Aberration]];

clearvars -except FileName PathName WR WR_full M_full Sheet_full X1 Q M X R ConicConst_k Mass Temperature
t = 1000;
script_RayTracing_Infinity_BFL
xlswrite(strcat(PathName,FileName),{'Backward Focal Length_With Glass Plate'},'RayTracing Results','A11')
A = {'Temperature(deg C)','Mass(mg)','R(micron)','k','Glass Plate Thickness_t(micron)','Paraxial Focal Length(micron)', ...
    'Surface of Least Confusion(micron)','Spot Size_dia(micron)','Longitudinal Spherical Aberration_max(micron)', ...
    'Longitudinal Spherical Aberration Coefficient','Transverse Spherical Aberration_max(micron)', ...
    'Transverse Spherical Aberration Coefficient','Longitudinal Chromatic Aberration(micron)';Temperature,Mass,R,ConicConst_k,t,Paraxial_focal_length, ...
    z_lcs,beam_t_min,LSA_max,a_lsa,TSA_max,a_tsa,Longitudinal_Chromatic_Aberration};
xlswrite(strcat(PathName,FileName),A,'RayTracing Results','AN2')

WR = [WR [Temperature Mass R ConicConst_k t Paraxial_focal_length ...
    z_lcs beam_t_min LSA_max a_lsa TSA_max a_tsa Longitudinal_Chromatic_Aberration]];


clearvars -except FileName PathName X1 WR WR_full M_full Sheet_full Q M X R ConicConst_k Mass Temperature
t = 0;
script_RayTracing_Magnification
xlswrite(strcat(PathName,FileName),{'Magnification(X)'},'RayTracing Results','A15')
A = {'Magnification(X)';Magnification};
xlswrite(strcat(PathName,FileName),A,'RayTracing Results','BA2')

WR = [WR Magnification];
