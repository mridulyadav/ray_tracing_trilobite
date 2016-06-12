

X; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0
R; %Radius of curvature at the vertex
ConicConst_k;
size_X = size(X);

syms r
z_exp = -(((r^2)/R)/(1+sqrt(1-(1+ConicConst_k)*((r^2)/(R^2)))));

z_func = @(r) subs(z_exp, 'r', r);

z_prime = diff(z_exp);

z_prime_func = @(r) subs(z_prime, 'r', r);




%Ensuring the profile is symmetric about the vertex
zero_index_1 = 0;
zero_index_2 = 0;
flag1 = 0;
flag2 = 0;
for i = 1:1:size_X(1)
    if((X(i,2)==0))
        zero_index_1 = i;
        break;
    end
end

for i = size_X(1):-1:1
    if((X(i,2)==0))
        zero_index_2 = i;
        break;
    end
end
zero_index = int32((zero_index_1 + zero_index_2)/2);



%   Centering data such that at vertex(z=0), x is 0
x_center = X(zero_index,1);
X(:,1) = X(:,1) - x_center;



X; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0

z = -X(:,2);% -ve curvature
x = -X(:,1);%


%%%Drawing the lens
f1 = figure
plot(z,x,'Color','k','LineWidth',1.5);

%%%Plane side of lens
[z_min,z_min_index] = min(z);

Z_planeInt = [z(z_min_index) z(z_min_index)];

if(z_min_index<zero_index)
    X_planeInt = [x(z_min_index) x(end)];
elseif(z_min_index>zero_index)
    X_planeInt = [x(z_min_index) x(1)];
end

line(Z_planeInt,X_planeInt,'Color','k','LineWidth',1.5);


%%%Drawing Optical Axis/Co-ordinate system
%Z-Axis
z_optical_axis_min = -100000;
z_optical_axis_max = 100000;
Z_optical_axis_z = [z_optical_axis_min z_optical_axis_max];
Z_optical_axis_x = [0 0];
line(Z_optical_axis_z,Z_optical_axis_x,'Color','k','LineWidth',1)

%X-Axis
x_optical_axis_min = -5000;
x_optical_axis_max = 5000;
X_optical_axis_z = [0 0];
X_optical_axis_x = [x_optical_axis_min x_optical_axis_max];
line(X_optical_axis_z,X_optical_axis_x,'Color','k','LineWidth',1)


%%%Drawing the glass plate
%microns : Thickness of glass plate
%Drawing the interface
[z_min,z_min_index] = min(z);

Z_air_glass_Int = [(z(z_min_index)-t) (z(z_min_index)-t)];

if(z_min_index<zero_index)
    X_air_glass_Int = [x(z_min_index) x(end)];
elseif(z_min_index>zero_index)
    X_air_glass_Int = [x(z_min_index) x(1)];
end

line(Z_air_glass_Int,X_air_glass_Int,'Color','k','LineWidth',1.5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Number of rays
n = 10;

%%Ray Parameters : Each ray has a starting point and an ending point and
%%angle with z axis;
ray_initial_up = zeros(n,2);
ray_final_up = zeros(n,2);
ray_angle_up = zeros(n,1);
ray_initial_down = zeros(n,2);
ray_final_down = zeros(n,2);
ray_angle_down = zeros(n,1);

%%% Paraxial Rays
paraxial_angle = 1;%in degrees
min_height = R*tand(paraxial_angle);%in microns

diff = abs(min_height - x(1:zero_index,1));
[min_diff,index_min_height] = min(diff);

c1 = abs(X(1,1));
c2 = abs(X(end,1));

c3 = min(c1,c2);%maximum height i.e. height of final ray; this height is divided into n equal parts
max_height = c3-100;

%Point Source at infinity
%Ray origin for display
z_ray_origin = -5000;
for i = 1:n
    ray_initial_up(i,1) = z_ray_origin;
    ray_initial_up(i,2) = min_height + ((max_height-min_height)/(n-1))*(i-1);
    ray_initial_down(i,1) = z_ray_origin;
    ray_initial_down(i,2) = -min_height-((max_height-min_height)/(n-1))*(i-1);
end


h = ray_initial_up(:,2); % for LSA

for i = 1:n
    ray_i_x_up = min_height + ((max_height-min_height)/(n-1))*(i-1);
    ray_final_up(i,1) = z_func(ray_i_x_up);
    ray_final_up(i,2) = ray_i_x_up;

    ray_i_x_down = -min_height-((max_height-min_height)/(n-1))*(i-1);
    ray_final_down(i,1) = z_func(ray_i_x_down);
    ray_final_down(i,2) = ray_i_x_down;

end
%2 : x
%1 : z
% Angles
alpha_left_up = [];
alpha_left_down = [];
for i = 1:n
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)));
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)));
end

n_left = 1.435;%lambda = 532 nm
n_right = 1;

%%% Checking for maximum angle before TIR
flag_TIR = 0;
m_tangent = 0;
k_max_TIR_down = 1;
k_max_TIR_down = size_X(1);
x_test = 0;

while (flag_TIR==0)
    flag_TIR_up = 0;
    flag_TIR_down = 0;
    flag_TIR = 0;
    for x_test = min_height:1:max_height
        %above optical axis
        %k = ray_final_index_up(i,1);
        
        
        x_test;
        
        m_tangent = z_prime_func(x_test);
        phi = 90 - abs(atand(double(m_tangent)));
      
        alpha_left_up_test = 0;
        
        
        if(alpha_left_up(i,1)>=0)
            theta_left = abs(90 - alpha_left_up_test - phi);
        elseif(alpha_left_up(i,1)<0)
            theta_left = abs(90 + alpha_left_up_test - phi);
        end
        
        if(sind(theta_left)>= (n_right/n_left))
            flag_TIR_up = 1;
            x_test_max_up = x_test;
            break;
        end
        
        
    end
    
    
    for x_test = -min_height:-1:-max_height
        %below optical axis
        %k = ray_final_index_down(i,1);
        m_tangent = z_prime_func(x_test);
        phi = 90 - abs(atand(double(m_tangent)));
        
        alpha_left_down_test = 0;
        
        if(alpha_left_down(i,1)>=0)
            theta_left = abs(90 - alpha_left_down_test - phi);
        elseif(alpha_left_down(i,1)<0)
            theta_left = abs(90 + alpha_left_down_test - phi);
        end
        
        if(sind(theta_left)>= (n_right/n_left))
            flag_TIR_down = 1;
            x_test_max_down = abs(x_test);
            break;
        end
        
    end
    
    if(flag_TIR_up == 1)||(flag_TIR_down == 1)
        if(flag_TIR_up == 1)
            max_height = (x_test_max_up) - 10;
        elseif(flag_TIR_down == 1)
            max_height = (x_test_max_down) - 10;
        elseif((flag_TIR_up == 1)&&(flag_TIR_down == 1))
            max_height = min(x_test_max_up,x_test_max_down) - 10;
        end
        flag_TIR = 0;
    else
        flag_TIR = 1;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Point Source at Infinity %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ray origin for display
z_ray_origin = -5000;
for i = 1:n
    ray_initial_up(i,1) = z_ray_origin;
    ray_initial_up(i,2) = min_height + ((max_height-min_height)/(n-1))*(i-1);
    ray_initial_down(i,1) = z_ray_origin;
    ray_initial_down(i,2) = -min_height-((max_height-min_height)/(n-1))*(i-1);
end

% Angles
alpha_left_up = zeros(n,1);
alpha_left_down = zeros(n,1);
for i = 1:n
    alpha_left_up(i,1) = 0;
    alpha_left_down(i,1) = 0;
end

alpha_right_up = alpha_left_up;
alpha_right_down = alpha_left_down;

color = [0.75 0.75 0.75];

z_final = z(z_min_index) - t;
m_up = tand(alpha_left_up);
m_down = tand(alpha_left_down);

%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Interface 1 : Air-Glass Interface %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

%   Refraction
n_left = 1;
n_right = 1.52;

m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down(i,1) = theta_right;
    
end

z_final = z(z_min_index);
m_up = tand(alpha_right_up);
m_down = tand(alpha_right_down);
%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Interface 2 : Glass-PDMS %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

% Refraction
n_left = 1.52;
n_right = 1.435;%lambda = 532 nm

m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down(i,1) = theta_right;
    
end

%   Calculating the final z-coordinate for the PDMS-Air Interface


m_up = tand(alpha_right_up);
m_down = tand(alpha_right_down);

ray_final_index_up = zeros(n,1);
ray_final_index_down = zeros(n,1);
    
    for i = 1:n
    ray_i_x_up = ray_initial_up(i,2);
    
%     diff = abs(ray_i_x_up - x(1:zero_index,1));
%     [min_diff,index_ray_i_x] = min(diff);
    
    ray_final_up(i,1) = z_func(ray_i_x_up);
    ray_final_up(i,2) = ray_i_x_up;
%    ray_final_index_up(i,1) = index_ray_i_x;
    

    ray_i_x_down =ray_initial_down(i,2);
%     diff = abs(ray_i_x_down - x((zero_index):end,1))
%     [min_diff,index_ray_i_x] = min(diff)
    
    ray_final_down(i,1) = z_func(ray_i_x_down);
    ray_final_down(i,2) = ray_i_x_down;
 %   ray_final_index_down(i,1) = (zero_index-1)+index_ray_i_x;
end
    
    

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);
%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Interface 3 : PDMS-Air Interface %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

%%%Refraction
n_left = 1.435;%lambda = 532 nm
n_right = 1;
%m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    %k = ray_final_index_up(i,1);
    m_tangent = z_prime_func(ray_final_up(i,2));
    phi = 90 - abs(atand(double(m_tangent)));
    
    
    if(alpha_left_up(i,1)>=0)
        theta_left = abs(90 - alpha_left_up(i,1) - phi);
    elseif(alpha_left_up(i,1)<0)
        theta_left = abs(90 + alpha_left_up(i,1) - phi);
    end
    
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_up(i,1) = 90 - phi - theta_right;
    
    %below optical axis
   % k = ray_final_index_down(i,1);
    m_tangent = z_prime_func(ray_final_down(i,2));
    phi = 90 - abs(atand(double(m_tangent)));
    
    if(alpha_left_down(i,1)>=0)
        theta_left = abs(90 - alpha_left_down(i,1) - phi);
    elseif(alpha_left_down(i,1)<0)
        theta_left = abs(90 + alpha_left_down(i,1) - phi);
    end
    
    theta_right = abs(asind((n_left/n_right)*sind(theta_left)));
    
    alpha_right_down(i,1) = -90 + phi + theta_right;
    
    
end




%%%Right of Interface 1
z_final = 50000;
m_up = tand(alpha_right_up);
m_down = tand(alpha_right_down);
%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

figname = ['FFL_RayDiagram_t_' num2str(t)];
saveas(f1,strcat(PathName,figname),'fig');

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);




%%%Calculating Focal Length and Longitudinal Spherical Aberration(LSA)

m1 = tand(alpha_left_up);
c1 = ray_initial_up(:,2) - m1.*ray_initial_up(:,1);

m2 = tand(alpha_left_down);
c2 = ray_initial_down(:,2) - m2.*ray_initial_down(:,1);

z_intersection = (c2-c1)./(m1-m2);

Paraxial_focal_length = z_intersection(1,1);
Meridinal_focal_length = z_intersection(n,1);

LSA_max = Paraxial_focal_length - Meridinal_focal_length;

LSA = Paraxial_focal_length - z_intersection;
y_lsa = LSA(1:end) - LSA(1);
x_lsa = h(1:end) - h(1);

g_lsa = fittype( @(a,x) a*x.^2);
[f_lsa,gof_lsa] = fit(x_lsa,y_lsa,g_lsa);
a_lsa = f_lsa.a;
f2 = figure;
plot(f_lsa,x_lsa,y_lsa);
figname = ['FFL_LongitudinalSA_vs_Height_t_' num2str(t)];
saveas(f2,strcat(PathName,figname),'fig');

%%% Transverse Spherical Aberration
TSA_up = abs((m2*z_intersection(1,1) + c2) - (m2(1,1)*z_intersection(1,1) + c2(1,1)));
TSA_down = abs((m1*z_intersection(1,1) + c1) - (m1(1,1)*z_intersection(1,1) + c1(1,1)));

TSA_avg = (TSA_up+TSA_down)/2;

TSA_max = max(TSA_avg);

y_tsa = TSA_avg(1:end) - TSA_avg(1);
x_tsa = h(1:end) - h(1);

g_tsa = fittype( @(a,x) a*x.^3);
[f_tsa,gof_tsa] = fit(x_tsa,y_tsa,g_tsa)
a_tsa = f_tsa.a
f3 = figure
plot(f_tsa,x_tsa,y_tsa)
figname = ['FFL_TransverseSA_vs_Height_t_' num2str(t)];
saveas(f3,strcat(PathName,figname),'fig');

%%%Calculating Surface of Least Confusion
m_slope_up = [];
m_slope_down = [];
min_lcs = Inf;
z_lcs = 0;
beam_t_min = 0;
m_slope_up = tand(alpha_left_up);
m_slope_down = tand(alpha_left_down);
for z = 0:0.1:Paraxial_focal_length
    
    y_up = m_slope_up*z + (ray_initial_up(:,2) - m_slope_up.*ray_initial_up(:,1));
    y_down = m_slope_down*z + (ray_initial_down(:,2) - m_slope_down.*ray_initial_down(:,1));
    
    y_all = [y_up; y_down];
    
    beam_t = max(y_all) - min(y_all);
    
    if(beam_t<min_lcs)
        z_lcs = z;
        min_lcs = beam_t;
        beam_t_min = beam_t;
    end
    
    
end

z_lcs;
min_lcs;
beam_t_min;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ____ _                               _   _           _    _                         _   _                
% / ___| |__  _ __ ___  _ __ ___   __ _| |_(_) ___     / \  | |__   ___ _ __ _ __ __ _| |_(_) ___  _ __  
%| |   | '_ \| '__/ _ \| '_ ` _ \ / _` | __| |/ __|   / _ \ | '_ \ / _ \ '__| '__/ _` | __| |/ _ \| '_ \ 
%| |___| | | | | | (_) | | | | | | (_| | |_| | (__   / ___ \| |_) |  __/ |  | | | (_| | |_| | (_) | | | |
% \____|_| |_|_|  \___/|_| |_| |_|\__,_|\__|_|\___| /_/   \_\_.__/ \___|_|  |_|  \__,_|\__|_|\___/|_| |_|
%                                                                                                        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars -except FileName PathName Mass WR WR_full M_full Sheet_full Temperature X R ConicConst_k t max_height min_height Paraxial_focal_length LSA_max y_lsa x_lsa a_lsa TSA_max y_tsa x_tsa a_tsa z_lcs min_lcs beam_t_min;

syms r
z_exp = -(((r^2)/R)/(1+sqrt(1-(1+ConicConst_k)*((r^2)/(R^2)))));

z_func = @(r) subs(z_exp, 'r', r);

z_prime = diff(z_exp);

z_prime_func = @(r) subs(z_prime, 'r', r);

X; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0
R; %Radius of curvature at the vertex
size_X = size(X);


%Ensuring the profile is symmetric about the vertex
zero_index_1 = 0;
zero_index_2 = 0;
flag1 = 0;
flag2 = 0;
for i = 1:1:size_X(1)
    if((X(i,2)==0))
        zero_index_1 = i;
        break;
    end
end

for i = size_X(1):-1:1
    if((X(i,2)==0))
        zero_index_2 = i;
        break;
    end
end
zero_index = int32((zero_index_1 + zero_index_2)/2);



%   Centering data such that at vertex(z=0), x is 0
x_center = X(zero_index,1);
X(:,1) = X(:,1) - x_center;



X; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0

z = -X(:,2);% -ve curvature
x = -X(:,1);%


%%%Drawing the lens
f_chromatic = figure;
plot(z,x,'Color','k','LineWidth',1.5);

%%%Plane side of lens
[z_min,z_min_index] = min(z);

Z_planeInt = [z(z_min_index) z(z_min_index)];

if(z_min_index<zero_index)
    X_planeInt = [x(z_min_index) x(end)];
elseif(z_min_index>zero_index)
    X_planeInt = [x(z_min_index) x(1)];
end

line(Z_planeInt,X_planeInt,'Color','k','LineWidth',1.5);


%%%Drawing Optical Axis/Co-ordinate system
%Z-Axis
z_optical_axis_min = -100000;
z_optical_axis_max = 100000;
Z_optical_axis_z = [z_optical_axis_min z_optical_axis_max];
Z_optical_axis_x = [0 0];
line(Z_optical_axis_z,Z_optical_axis_x,'Color','k','LineWidth',1)

%X-Axis
x_optical_axis_min = -5000;
x_optical_axis_max = 5000;
X_optical_axis_z = [0 0];
X_optical_axis_x = [x_optical_axis_min x_optical_axis_max];
line(X_optical_axis_z,X_optical_axis_x,'Color','k','LineWidth',1)


%%%Drawing the glass plate

%Drawing the interface
[z_min,z_min_index] = min(z);

Z_air_glass_Int = [(z(z_min_index)-t) (z(z_min_index)-t)];

if(z_min_index<zero_index)
    X_air_glass_Int = [x(z_min_index) x(end)];
elseif(z_min_index>zero_index)
    X_air_glass_Int = [x(z_min_index) x(1)];
end

line(Z_air_glass_Int,X_air_glass_Int,'Color','k','LineWidth',1.5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Number of rays
n = 1;

%%Ray Parameters : Each ray has a starting point and an ending point and
%%angle with z axis;
ray_initial_up = zeros(n,2);
ray_final_up = zeros(n,2);
ray_angle_up = zeros(n,1);
ray_initial_down = zeros(n,2);
ray_final_down = zeros(n,2);
ray_angle_down = zeros(n,1);

%Point Source at infinity
%Ray origin for display
z_ray_origin = -5000;
for i = 1:n
    ray_initial_up(i,1) = z_ray_origin;
    ray_initial_up(i,2) = min_height + ((max_height-min_height)/(n))*(i-1);
    ray_initial_down(i,1) = z_ray_origin;
    ray_initial_down(i,2) = -min_height-((max_height-min_height)/(n))*(i-1);
end

h = ray_initial_up(:,2); % for LSA

  for i = 1:n
    ray_i_x_up = min_height + ((max_height-min_height)/(n))*(i-1);
    
%     diff = abs(ray_i_x_up - x(1:zero_index,1));
%     [min_diff,index_ray_i_x] = min(diff);
    
    ray_final_up(i,1) = z_func(ray_i_x_up);
    ray_final_up(i,2) = ray_i_x_up;
%    ray_final_index_up(i,1) = index_ray_i_x;
    

    ray_i_x_down =-min_height-((max_height-min_height)/(n))*(i-1);
%     diff = abs(ray_i_x_down - x((zero_index):end,1))
%     [min_diff,index_ray_i_x] = min(diff)
    
    ray_final_down(i,1) = z_func(ray_i_x_down);
    ray_final_down(i,2) = ray_i_x_down;
 %   ray_final_index_down(i,1) = (zero_index-1)+index_ray_i_x;
end

%2 : x
%1 : z
% Angles
alpha_left_up = [];
alpha_left_down = [];
for i = 1:n
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)));
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Point Source at Infinity %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ray origin for display
z_ray_origin = -5000;
for i = 1:n
    ray_initial_up(i,1) = z_ray_origin;
    ray_initial_up(i,2) = min_height + ((max_height-min_height)/(n))*(i-1);
    ray_initial_down(i,1) = z_ray_origin;
    ray_initial_down(i,2) = -min_height-((max_height-min_height)/(n))*(i-1);
end

% Angles
alpha_left_up = zeros(n,1);
alpha_left_down = zeros(n,1);
for i = 1:n
    alpha_left_up(i,1) = 0;
    alpha_left_down(i,1) = 0;
end

alpha_right_up = alpha_left_up;
alpha_right_down = alpha_left_down;

color = [0.75 0.75 0.75];

z_final = z(z_min_index) - t;
m_up = tand(alpha_left_up);
m_down = tand(alpha_left_down);

%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Interface 1 : Air-Glass Interface %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

%   Refraction
n_left = 1;
n_right = 1.52;

m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down(i,1) = theta_right;
    
end

z_final = z(z_min_index);
m_up = tand(alpha_right_up);
m_down = tand(alpha_right_down);
%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);

ray_initial_up_orignl = ray_final_up;
ray_initial_down_orignl = ray_final_down;
alpha_left_up_orignl = alpha_right_up;
alpha_left_down_orignl = alpha_right_down;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Interface 2 : Glass-PDMS %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ctr = 1:3
    if (ctr==1)% BLUE
        n_color = 1.448;
        color = 'b';
    elseif(ctr==2)
        n_color = 1.429;
        color = 'r';
    elseif(ctr==3)
        n_color = 1.435;
        color = 'y';
    end
    
    ray_initial_up = ray_initial_up_orignl;
    ray_initial_down = ray_initial_down_orignl;
    alpha_left_up = alpha_left_up_orignl;
    alpha_left_down = alpha_left_down_orignl;
    
    % Refraction
    n_left = 1.52;
    n_right = n_color;%lambda = 405 nm
    
    m_tangent = zeros(n,1);
    alpha_right_up = zeros(n,1);
    alpha_right_down = zeros(n,1);
    for i = 1:n
        %above optical axis
        theta_left = alpha_left_up(i,1);
        theta_right = asind((n_left/n_right)*sind(theta_left));
        
        alpha_right_up(i,1) = theta_right;
        
        %below optical axis
        theta_left = alpha_left_down(i,1);
        theta_right = asind((n_left/n_right)*sind(theta_left));
        
        alpha_right_down(i,1) = theta_right;
        
    end
    
    %   Calculating the final z-coordinate for the PDMS-Air Interface
    
    
    m_up = tand(alpha_right_up);
    m_down = tand(alpha_right_down);
    
    ray_final_index_up = zeros(n,1);
    ray_final_index_down = zeros(n,1);

    
    for i = 1:n
        ray_i_x_up = ray_initial_up(i,2);
        
        %     diff = abs(ray_i_x_up - x(1:zero_index,1));
        %     [min_diff,index_ray_i_x] = min(diff);
        
        ray_final_up(i,1) = z_func(ray_i_x_up);
        ray_final_up(i,2) = ray_i_x_up;
        %    ray_final_index_up(i,1) = index_ray_i_x;
        
        
        ray_i_x_down = ray_initial_down(i,2);
        %     diff = abs(ray_i_x_down - x((zero_index):end,1))
        %     [min_diff,index_ray_i_x] = min(diff)
        
        ray_final_down(i,1) = z_func(ray_i_x_down);
        ray_final_down(i,2) = ray_i_x_down;
        %   ray_final_index_down(i,1) = (zero_index-1)+index_ray_i_x;
    end
   
   
    %   Tracing Rays
    TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);
    
    %   Angles
    [alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% Interface 3 : PDMS-Air Interface %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ray_initial_up = ray_final_up;
    ray_initial_down = ray_final_down;
    alpha_left_up = alpha_right_up;
    alpha_left_down = alpha_right_down;
    
    %%%Refraction
    n_left = n_color;
    n_right = 1;
    m_tangent = zeros(n,1);
    alpha_right_up = zeros(n,1);
    alpha_right_down = zeros(n,1);
    for i = 1:n
        %above optical axis
        %k = ray_final_index_up(i,1);
        m_tangent = z_prime_func(ray_final_up(i,2));
        phi = 90 - abs(atand(double(m_tangent)));
        
        
        if(alpha_left_up(i,1)>=0)
            theta_left = abs(90 - alpha_left_up(i,1) - phi);
        elseif(alpha_left_up(i,1)<0)
            theta_left = abs(90 + alpha_left_up(i,1) - phi);
        end
        
        theta_right = asind((n_left/n_right)*sind(theta_left));
        
        alpha_right_up(i,1) = 90 - phi - theta_right;
        
        %below optical axis
        % k = ray_final_index_down(i,1);
        m_tangent = z_prime_func(ray_final_down(i,2));
        phi = 90 - abs(atand(double(m_tangent)));
        
        if(alpha_left_down(i,1)>=0)
            theta_left = abs(90 - alpha_left_down(i,1) - phi);
        elseif(alpha_left_down(i,1)<0)
            theta_left = abs(90 + alpha_left_down(i,1) - phi);
        end
        
        theta_right = abs(asind((n_left/n_right)*sind(theta_left)));
        
        alpha_right_down(i,1) = -90 + phi + theta_right;
        
        
    end
    
    
    
    
    
    %%%Right of Interface 1
    z_final = 50000;
    m_up = tand(alpha_right_up);
    m_down = tand(alpha_right_down);
    %   Calculating final co-ordinates of rays
    [ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);
    
    %   Tracing Rays
    TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color)
    
    %   Angles
    [alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);
    
    m1 = tand(alpha_left_up);
    c1 = ray_initial_up(:,2) - m1.*ray_initial_up(:,1);
    
    m2 = tand(alpha_left_down);
    c2 = ray_initial_down(:,2) - m2.*ray_initial_down(:,1);
    
    z_intersection = (c2-c1)./(m1-m2);
    
    Paraxial_focal_length = z_intersection(1,1);
    if (ctr==1)
        Parax_focal_length_blue = Paraxial_focal_length;
    elseif(ctr==2)
        Parax_focal_length_red = Paraxial_focal_length;
    end
    
end

figname = ['FFL_Chromatic_Aberration_t_' num2str(t)];
saveas(f_chromatic,strcat(PathName,figname),'fig');

Longitudinal_Chromatic_Aberration = abs(Parax_focal_length_red - Parax_focal_length_blue);

