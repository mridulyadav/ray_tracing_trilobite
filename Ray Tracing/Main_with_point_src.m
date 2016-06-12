
clear

[FileName2,PathName2] = uigetfile('*.*','Select Input File');
[status,sheets] = xlsfinfo(strcat(PathName2,FileName2));

[s,v] = listdlg('PromptString','Select a Sheet Name:',...
    'SelectionMode','single',...
    'ListString',sheets);
Sheet_temp = sheets(s);
Sheet = Sheet_temp{1};

flag_light_source = 0;
% Construct a questdlg with three options
choice = questdlg('Select position of light source on optical axis:', ...
    'Light source position', ...
    'Infinity','Finite Distance','Infinity');
% Handle response
switch choice
    case 'Infinity'
        flag_light_source = 1;
    case 'Finite Distance'
        flag_light_source = 2;
end

% Input when light source is at Infinity
if(flag_light_source == 1)
    prompt = {'Film Thickness (micron) :','Aperture Diameter (micron) :',...
        'Number of rays :', 'Refractive Index (PDMS) :', 'Paraxial Angle (degrees) :'};
    dlg_title = 'Input';
    num_lines = 1;
    defaultans = {'600','2000','15','1.435','1'};
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    t_film = str2double(answer{1});
    d_aperture = str2double(answer{2});
    %Number of rays = 2*n
    n = str2double(answer{3});
    n_PDMS = str2double(answer{4});
    paraxial_angle = str2double(answer{5});%in degrees
    
    % Input when light source is at a Finite distance
elseif(flag_light_source == 2)
    prompt = {'Film Thickness (micron) :','Aperture Diameter (micron) :',...
        'Number of rays :', 'Refractive Index (PDMS) :', 'Paraxial Angle (degrees) :','Position of Point Source (micron):'};
    dlg_title = 'Input';
    num_lines = 1;
    defaultans = {'600','2000','15','1.435','1','-5000'};
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    t_film = str2double(answer{1});
    d_aperture = str2double(answer{2});
    %Number of rays = 2*n
    n = str2double(answer{3});
    n_PDMS = str2double(answer{4});
    paraxial_angle = str2double(answer{5});%in degrees
    point_src_location = str2double(answer{6});
else
    disp('No option for light source selected.')
    return;
end



%% Reading profile information from the input file
% Reading profile data for top lens
X_top_1 = xlsread(strcat(PathName2,FileName2),Sheet,'C:E');
X_top = [X_top_1(:,1) X_top_1(:,2)];
R_top = xlsread(strcat(PathName2,FileName2),Sheet,'B5');
ConicConst_k_top = xlsread(strcat(PathName2,FileName2),Sheet,'B6');

% Reading profile data for bottom lens
X_bot_1 = xlsread(strcat(PathName2,FileName2),Sheet,'I:K');
X_bot = [X_bot_1(:,1) X_bot_1(:,2)];
R_bot = xlsread(strcat(PathName2,FileName2),Sheet,'H5');
ConicConst_k_bot = xlsread(strcat(PathName2,FileName2),Sheet,'H6');

z_top = X_top(:,2);% -ve curvature
x_top = X_top(:,1);
size_X_top = size(X_top);

A4_bot = 0;
z_bot = X_bot(:,2);% -ve curvature
x_bot = X_bot(:,1);
size_X_bot = size(X_bot);

%% Converting the Input data to symbolic functions to calculate derivatives.
syms r
z_exp_top = (((r^2)/R_top)/(1+sqrt(1-(1+ConicConst_k_top)*((r^2)/(R_top^2)))));
z_func_top = @(r) subs(z_exp_top, 'r', r);
z_prime_top = diff(z_exp_top);
z_prime_func_top = @(r) subs(z_prime_top, 'r', r);

z_exp_bot = -((((r^2)/R_bot)/(1+sqrt(1-(1+ConicConst_k_bot)*((r^2)/(R_bot^2))))) + A4_bot*(r^4));
z_func_bot = @(r) subs(z_exp_bot, 'r', r);
z_prime_bot = diff(z_exp_bot);
z_prime_func_bot = @(r) subs(z_prime_bot, 'r', r);

%% Ensuring the profile is symmetric about the vertex
zero_index_1 = 0;
zero_index_2 = 0;
for i = 1:1:size_X_top(1)
    if((X_top(i,2)==0))
        zero_index_1 = i;
        break;
    end
end

for i = size_X_top(1):-1:1
    if((X_top(i,2)==0))
        zero_index_2 = i;
        break;
    end
end
zero_index_top = int32((zero_index_1 + zero_index_2)/2);

zero_index_1 = 0;
zero_index_2 = 0;
for i = 1:1:size_X_bot(1)
    if((X_bot(i,2)==0))
        zero_index_1 = i;
        break;
    end
end

for i = size_X_bot(1):-1:1
    if((X_bot(i,2)==0))
        zero_index_2 = i;
        break;
    end
end
zero_index_bot = int32((zero_index_1 + zero_index_2)/2);

%% Centering data such that at vertex(z=0), x is 0
x_center_top = X_top(zero_index_top,1);
X_top(:,1) = X_top(:,1) - x_center_top;

x_center_bot = X_bot(zero_index_bot,1);
X_bot(:,1) = X_bot(:,1) - x_center_bot;

%% Lens Dimensions
t1 = max(abs(z_top(1:zero_index_top)));
t2 = max(abs(z_top(zero_index_top:end)));
thickness_top_lens = min(t1,t2);

t1 = max(abs(z_bot(1:zero_index_bot)));
t2 = max(abs(z_bot(zero_index_bot:end)));
thickness_bot_lens = min(t1,t2);

w1 = max(abs(x_top(1:zero_index_top)));
w2 = max(abs(x_top(zero_index_top:end)));
width_top_lens = 2*min(w1,w2);

w1 = max(abs(x_bot(1:zero_index_bot)));
w2 = max(abs(x_bot(zero_index_bot:end)));
width_bot_lens = 2*min(w1,w2);

%% If aperture size is larger than width of top lens
if(d_aperture>=width_top_lens)
    choice = questdlg('Aperture diameter is larger than diameter of top lens. Continue with aperture diameter = diameter of top lens?', ...
        'Error', ...
        'Continue','Stop','Continue');
    % Handle response
    switch choice
        case 'Continue'
            d_aperture = width_top_lens;
        case 'Stop'
            return;
    end
    
end

%% Drawing the lens
X_top; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0
z_top = X_top(:,2) - thickness_top_lens - thickness_bot_lens - t_film;% -ve curvature
x_top = X_top(:,1);%

X_bot; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0
z_bot = -X_bot(:,2);% -ve curvature
x_bot = X_bot(:,1);%

f1 = figure;
hold on
plot(z_top,x_top,'Color','k','LineWidth',1.5);
plot(z_bot,x_bot,'Color','k','LineWidth',1.5);

%% Drawing the film
film_width = max(width_top_lens,width_bot_lens) + 1000;

film_bot_z = [-thickness_bot_lens -thickness_bot_lens];
film_bot_x = [film_width/2 -film_width/2];
line(film_bot_z,film_bot_x,'Color','k')

film_top_z = [-thickness_bot_lens-t_film -thickness_bot_lens-t_film];
film_top_x = [film_width/2 -film_width/2];
line(film_top_z,film_top_x,'Color','k')

%% Drawing the aperture
aperture_top_z = [(-thickness_bot_lens-(t_film/2)) (-thickness_bot_lens-(t_film/2))];
aperture_top_x = [d_aperture/2 film_width/2];
line(aperture_top_z,aperture_top_x,'Color','b','LineWidth',2,'LineStyle',':')

aperture_bot_z = [(-thickness_bot_lens-(t_film/2)) (-thickness_bot_lens-(t_film/2))];
aperture_bot_x = [-d_aperture/2 -film_width/2];
line(aperture_bot_z,aperture_bot_x,'Color','b','LineWidth',2,'LineStyle',':')

%% Ray Parameters : Each ray has a starting point and an ending point and
%angle with z axis;
ray_initial_up = zeros(n,2);
ray_final_up = zeros(n,2);
ray_angle_up = zeros(n,1);
ray_initial_down = zeros(n,2);
ray_final_down = zeros(n,2);
ray_angle_down = zeros(n,1);

% Paraxial Rays
% Minimum height of rays
min_height = R_top*tand(paraxial_angle);

diff = abs(min_height - x_top(1:zero_index_top,1));
[min_diff,index_min_height] = min(diff);

% Maximum height of rays
max_height = d_aperture/2;

%% Incidence of rays from source

if(flag_light_source == 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% Point Source at Infinity %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %   Ray origin for display
    z_ray_origin = -10000;
    for i = 1:n
        ray_initial_up(i,1) = z_ray_origin;
        ray_initial_up(i,2) = min_height + ((max_height-min_height)/(n-1))*(i-1);
        ray_initial_down(i,1) = z_ray_origin;
        ray_initial_down(i,2) = -min_height-((max_height-min_height)/(n-1))*(i-1);
    end
    
    h = ray_initial_up(:,2);
    
    alpha_left_up = zeros(n,1);
    alpha_left_down = zeros(n,1);
    for i = 1:n
        alpha_left_up(i,1) = 0;
        alpha_left_down(i,1) = 0;
    end
    
    ray_final_index_up = zeros(n,1);
    ray_final_index_down = zeros(n,1);
    ray_final_up = zeros(n,1);
    ray_final_down = zeros(n,1);
    
    for i = 1:n
        ray_i_x_up = min_height + ((max_height-min_height)/(n-1))*(i-1);
        ray_final_up(i,1) = z_func_top(ray_i_x_up) - thickness_top_lens - thickness_bot_lens - t_film;
        ray_final_up(i,2) = ray_i_x_up;
        
        ray_i_x_down = -min_height-((max_height-min_height)/(n-1))*(i-1);
        ray_final_down(i,1) = z_func_top(ray_i_x_down) - thickness_top_lens - thickness_bot_lens - t_film;
        ray_final_down(i,2) = ray_i_x_down;
        
    end
    
    color = [0.5 0.5 0.5];
    
    % Tracing Rays
    TraceRays_func(1,n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);
    
    % Angles
    alpha_left_up = zeros(n,1);
    alpha_left_down = zeros(n,1);
    for i = 1:n
        alpha_left_up(i,1) = 0;
        alpha_left_down(i,1) = 0;
    end
    
    
elseif(flag_light_source == 2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Point Source at Finite Distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Ray origin for display
    z_point_source = point_src_location;
    x_point_source = 0;
    for i = 1:n
        ray_initial_up(i,1) = z_point_source;
        ray_initial_up(i,2) = x_point_source;
        ray_initial_down(i,1) = z_point_source;
        ray_initial_down(i,2) = x_point_source;
    end
    
    h = ray_initial_up(:,2); % for LSA
    
    ray_final_index_up = zeros(n,2);
    ray_final_index_down = zeros(n,2);
    ray_final_up = zeros(n,2);
    ray_final_down = zeros(n,2);
    
    %   Maximum and Minimum angles ; this range is divided in n parts
    alpha_min = atand(min_height/(abs(z_point_source)-abs(- thickness_top_lens - thickness_bot_lens - t_film)));
    alpha_max = atand(max_height/(abs(z_point_source)-abs(- thickness_top_lens - thickness_bot_lens - t_film)));
    
    alpha_left_up = zeros(n,1);
    alpha_left_down = zeros(n,1);
    for i = 1:n
        
        alpha_left_up(i,1) = alpha_min + ((alpha_max-alpha_min)/(n))*(i-1);
        alpha_left_down(i,1) = - alpha_left_up(i,1);
        
    end
    
    for i = 1:n
        %above optical axis
        m_up = tand(alpha_left_up(i,1));
        
        
        dist = sqrt((((z_top) - (1/m_up)*(x_top-(ray_initial_up(i,2)-m_up*ray_initial_up(i,1)))).^2)+((x_top - (m_up*z_top+(ray_initial_up(i,2)-m_up*ray_initial_up(i,1)))).^2));
        [~,index_min_lens_surface] = min(dist);
        
        ray_final_up(i,1) = z_top(index_min_lens_surface);
        ray_final_up(i,2) = x_top(index_min_lens_surface);
        ray_final_index_up(i,1) = index_min_lens_surface;
        
        %below optical axis
        m_down = tand(alpha_left_down(i,1));
        
        dist = sqrt(((z_top - (1/m_down)*(x_top-(ray_initial_down(i,2)-m_down*ray_initial_down(i,1)))).^2)+((x_top - (m_down*z_top+(ray_initial_down(i,2)-m_down*ray_initial_down(i,1)))).^2));
        [min_dist,index_min_lens_surface] = min(dist);
        
        
        ray_final_down(i,1) = z_top(index_min_lens_surface);
        ray_final_down(i,2) = x_top(index_min_lens_surface);
        ray_final_index_down(i,1) = index_min_lens_surface;
        
    end
    
    color = [0.5 0.5 0.5];
    
    % Tracing Rays
    TraceRays_func(1,n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);
    
else
    return;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Interface 1 : Air-PDMS Interface %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Refraction
n_left = 1;
n_right = n_PDMS;
m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    m_tangent = z_prime_func_top(ray_final_up(i,2));
    phi = (90-abs(atand(double(m_tangent))));
    
    
    if(alpha_left_up(i,1)>=0)
        theta_left = abs(90 - alpha_left_up(i,1) - phi);
    elseif(alpha_left_up(i,1)<0)
        theta_left = abs(90 + alpha_left_up(i,1) - phi);
    end
    
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up(i,1) = -(90 - phi - theta_right);
    
    %below optical axis
    m_tangent = z_prime_func_top(ray_final_down(i,2));
    phi = (90 - abs(atand(double(m_tangent))));
    
    if(alpha_left_down(i,1)>=0)
        theta_left = abs(90 - alpha_left_down(i,1) - phi);
    elseif(alpha_left_down(i,1)<0)
        theta_left = abs(90 + alpha_left_down(i,1) - phi);
    end
    
    theta_right = abs(asind((n_left/n_right)*sind(theta_left)));
    
    alpha_right_down(i,1) = -(-90 + phi + theta_right);
    
    
    
end

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

z_final = -thickness_bot_lens;
m_up = tand(alpha_left_up);
m_down = tand(alpha_left_down);

%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(1,n,z_final,m_up,m_down,ray_initial_up,ray_initial_down);

%   Tracing Rays : after refraction
TraceRays_func(1,n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(1,n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);


ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;


for i = 1:n
    if(ray_initial_up(i,2) > width_bot_lens/2)
        n_planar = i;
        break;
    elseif(max_height < (width_bot_lens/2))
        n_planar = n;
    end
end



n_left = n_PDMS;
n_right = 1;
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = n_planar:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down(i,1) = theta_right;
end

m_up = tand(alpha_right_up);
m_down = tand(alpha_right_down);

% %   Calculating final co-ordinates of rays
ray_final_up_planar = zeros(n-n_planar+1,2);
ray_final_down_planar = zeros(n-n_planar+1,2);

for i = n_planar:n
    ray_final_up_planar(i,2) = 0;
    ray_final_up_planar(i,1) = -(1/m_up(i,1))*(ray_initial_up(i,2) - m_up(i,1)*ray_initial_up(i,1));
    
    ray_final_down_planar(i,2) = 0;
    ray_final_down_planar(i,1) = -(1/m_down(i,1))*(ray_initial_down(i,2) - m_down(i,1)*ray_initial_down(i,1));
end

%   Tracing Rays : after refraction
if(n_planar ~= n)
    color = [0 0 1];
    TraceRays_func(n_planar,n,ray_initial_up,ray_initial_down,ray_final_up_planar,ray_final_down_planar,color);
end
color = [0.5 0.5 0.5];
%   Angles
[alpha_left_up_planar,alpha_left_down_planar] = AlphaLeft_func(n_planar,n,ray_initial_up,ray_initial_down,ray_final_up_planar,ray_final_down_planar);

ray_initial_up_nobotlens = ray_initial_up;
ray_initial_down_nobotlens = ray_initial_down;

ray_final_up_temp = zeros(n,2);
ray_final_down_temp = zeros(n,2);

n_end = n_planar;

if(n_planar~=n)
    n_end = n_planar - 1;
end

alpha_right_up_nobotlens = zeros(n_end,1);
alpha_right_down_nobotlens = zeros(n_end,1);

for i = 1:(n_end)
    
    m_slope_up = tand(alpha_left_up(i,1));
    c_intercept_up = ray_initial_up(i,2) - m_slope_up*ray_initial_up(i,1);
    x0 = ray_initial_up(i,1);
    options = optimoptions('fsolve','Display','none');
    x = fsolve(@(x) (-((((x^2)/R_bot)/(1+sqrt(1-(1+ConicConst_k_bot)*((x^2)/(R_bot^2))))) + A4_bot*(x^4)) - ((1/m_slope_up)*(x-c_intercept_up))),x0,options);
    
    ray_final_up_temp(i,2) = x;
    ray_final_up_temp(i,1) = z_func_bot(x);
    
    m_slope_down = tand(alpha_left_down(i,1));
    c_intercept_down = ray_initial_down(i,2) - m_slope_down*ray_initial_down(i,1);
    x0 = ray_initial_down(i,1);
    
    x = fsolve(@(x) (-((((x^2)/R_bot)/(1+sqrt(1-(1+ConicConst_k_bot)*((x^2)/(R_bot^2))))) + A4_bot*(x^4)) - ((1/m_slope_down)*(x-c_intercept_down))),x0);
    
    ray_final_down_temp(i,2) = x;
    ray_final_down_temp(i,1) = z_func_bot(x);
    
    
    % Assuming no Bottom lens
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up_nobotlens(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down_nobotlens(i,1) = theta_right;
    
end

ray_final_up_botlens = ray_final_up_temp;
ray_final_down_botlens = ray_final_down_temp;

%   Tracing Rays : after refraction
TraceRays_func(1,n_end,ray_initial_up,ray_initial_down,ray_final_up_botlens,ray_final_down_botlens,color);

ray_initial_up_botlens = ray_final_up_botlens;
ray_initial_down_botlens = ray_final_down_botlens;
alpha_left_up_botlens = alpha_left_up;
alpha_left_down_botlens = alpha_left_down;

% Refraction
n_left = n_PDMS;
n_right = 1;

alpha_right_up_botlens = zeros(n_end,1);
alpha_right_down_botlens = zeros(n_end,1);
for i = 1:n_end
    %above optical axis

    m_tangent = z_prime_func_bot(ray_final_up_botlens(i,2));
    phi = 90 - abs(atand(double(m_tangent)));
    
    
    if(alpha_left_up_botlens(i,1)>=0)
        theta_left = abs(90 - alpha_left_up(i,1) - phi);
    elseif(alpha_left_up_botlens(i,1)<0)
        theta_left = abs(90 + alpha_left_up(i,1) - phi);
    end
    
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_up_botlens(i,1) = 90 - phi - theta_right;
    
    %below optical axis

    m_tangent = z_prime_func_bot(ray_final_down_botlens(i,2));
    phi = 90 - abs(atand(double(m_tangent)));
    
    if(alpha_left_down_botlens(i,1)>=0)
        theta_left = abs(90 - alpha_left_down_botlens(i,1) - phi);
    elseif(alpha_left_down_botlens(i,1)<0)
        theta_left = abs(90 + alpha_left_down_botlens(i,1) - phi);
    end
    
    theta_right = abs(asind((n_left/n_right)*sind(theta_left)));
    
    alpha_right_down_botlens(i,1) = -90 + phi + theta_right;
    
    
end

% Right of Interface 1
m_up = tand(alpha_right_up_botlens);
m_down = tand(alpha_right_down_botlens);
% %   Calculating final co-ordinates of rays

for i = 1:n_end
    ray_final_up_botlens(i,2) = 0;
    ray_final_up_botlens(i,1) = -(1/m_up(i,1))*(ray_initial_up_botlens(i,2) - m_up(i,1)*ray_initial_up_botlens(i,1));
    
    ray_final_down_botlens(i,2) = 0;
    ray_final_down_botlens(i,1) = -(1/m_down(i,1))*(ray_initial_down_botlens(i,2) - m_down(i,1)*ray_initial_down_botlens(i,1));
end

% Tracing Rays
TraceRays_func(1,n_end,ray_initial_up_botlens,ray_initial_down_botlens,ray_final_up_botlens,ray_final_down_botlens,color);

% Angles
[alpha_left_up_botlens,alpha_left_down_botlens] = AlphaLeft_func(1,n_end,ray_initial_up_botlens,ray_initial_down_botlens,ray_final_up_botlens,ray_final_down_botlens);


m1_planar = tand(alpha_left_up_planar(n_planar:n,1));
c1_planar = ray_initial_up(n_planar:n,2) - m1_planar.*ray_initial_up(n_planar:n,1);

m2_planar = tand(alpha_left_down_planar(n_planar:n,1));
c2_planar = ray_initial_down(n_planar:n,2) - m2_planar.*ray_initial_down(n_planar:n,1);

z_intersection_planar = (c2_planar-c1_planar)./(m1_planar-m2_planar);

m1_botlens = tand(alpha_left_up_botlens(1:n_end,1));
c1_botlens = ray_initial_up_botlens(1:n_end,2) - m1_botlens.*ray_initial_up_botlens(1:n_end,1);

m2_botlens = tand(alpha_left_down_botlens(1:n_end,1));
c2_botlens = ray_initial_down_botlens(1:n_end,2) - m2_botlens.*ray_initial_down_botlens(1:n_end,1);

z_intersection_botlens = (c2_botlens-c1_botlens)./(m1_botlens-m2_botlens);

%% Drawing Optical Axis/Co-ordinate system
% Z-Axis
if(flag_light_source == 1)
    z_optical_axis_min = -15000;
    z_optical_axis_max = 15000;
elseif(flag_light_source == 2)
    z_optical_axis_min = point_src_location-1000;
    z_optical_axis_max = -point_src_location+1000;
end
Z_optical_axis_z = [z_optical_axis_min 1000+max([z_intersection_planar; z_intersection_botlens])];
Z_optical_axis_x = [0 0];
line(Z_optical_axis_z,Z_optical_axis_x,'Color','k')

xlim([z_optical_axis_min 1000+max([z_intersection_planar; z_intersection_botlens])]);
ylim(-film_bot_x);

% X-Axis
x_optical_axis_min = -5000;
x_optical_axis_max = 5000;
X_optical_axis_z = [0 0];
X_optical_axis_x = [x_optical_axis_min x_optical_axis_max];
line(X_optical_axis_z,X_optical_axis_x,'Color','k')
