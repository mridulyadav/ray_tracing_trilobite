


X; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0
R; %Radius of curvature at the vertex
z = X(:,2);% -ve curvature
x = X(:,1);

size_X = size(X);
color = [0.75 0.75 0.75];

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

z = X(:,2);% -ve curvature
x = -X(:,1);%




%%%Drawing the lens
f_magnification = figure
plot(z,x,'Color','k','LineWidth',1.5);

%%%Plane side of lens
[z_max,z_max_index] = max(z);

Z_planeInt = [z(z_max_index) z(z_max_index)];

if(z_max_index<zero_index)
    X_planeInt = [x(z_max_index) x(end)];
elseif(z_max_index>zero_index)
    X_planeInt = [x(z_max_index) x(1)];
end
    
line(Z_planeInt,X_planeInt,'Color','k','LineWidth',1.5);

%%%Drawing Optical Axis/Co-ordinate system
%Z-Axis 
z_optical_axis_min = -300000;
z_optical_axis_max = 10000;
Z_optical_axis_z = [z_optical_axis_min z_optical_axis_max];
Z_optical_axis_x = [0 0];
line(Z_optical_axis_z,Z_optical_axis_x,'Color','k')

%X-Axis 
x_optical_axis_min = -5000;
x_optical_axis_max = 5000;
X_optical_axis_z = [0 0];
X_optical_axis_x = [x_optical_axis_min x_optical_axis_max];
line(X_optical_axis_z,X_optical_axis_x,'Color','k')

t = 1000;%microns : Thickness of glass plate
%Drawing the interface
[z_max,z_max_index] = max(z);

Z_air_glass_Int = [(z(z_max_index)+t) (z(z_max_index)+t)];

if(z_max_index<zero_index)
    X_air_glass_Int = [x(z_max_index) x(end)];
elseif(z_max_index>zero_index)
    X_air_glass_Int = [x(z_max_index) x(1)];
end
    
line(Z_air_glass_Int,X_air_glass_Int,'Color','k','LineWidth',1.5);

color = [0.75 0.75 0.75];


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

%%% Paraxial Rays
paraxial_angle = 1;%in degrees
min_height = R*tand(paraxial_angle);

diff = abs(min_height - x(1:zero_index,1));
[min_diff,index_min_height] = min(diff);

c1 = abs(X(1,1));
c2 = abs(X(end,1));

c3 = min(c1,c2);%maximum height i.e. height of final ray; this height is divided into n equal parts
max_height = c3-50;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Point Source at Finite Distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Ray origin for display
z_point_source = -254000;
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
alpha_min = atand(min_height/abs(z(zero_index) + abs(z_point_source)));
alpha_max = atand(max_height/abs(z(z_max_index) + abs(z_point_source)));

for i = 1:n

alpha_left_up(i,1) = alpha_min + ((alpha_max-alpha_min)/(n))*(i-1);
alpha_left_down(i,1) = - alpha_left_up(i,1);

end

for i = 1:n
%above optical axis
m_up = tand(alpha_left_up(i,1));


dist = sqrt(((z(1:zero_index) - (1/m_up)*(x(1:zero_index)-(ray_initial_up(i,2)-m_up*ray_initial_up(i,1)))).^2)+((x(1:zero_index) - (m_up*z(1:zero_index)+(ray_initial_up(i,2)-m_up*ray_initial_up(i,1)))).^2));
[min_dist,index_min_lens_surface] = min(dist);

ray_final_up(i,1) = z(index_min_lens_surface);
ray_final_up(i,2) = x(index_min_lens_surface);
ray_final_index_up(i,1) = index_min_lens_surface;

%below optical axis
m_down = tand(alpha_left_down(i,1));

dist = sqrt(((z(zero_index:end) - (1/m_down)*(x(zero_index:end)-(ray_initial_down(i,2)-m_down*ray_initial_down(i,1)))).^2)+((x(zero_index:end) - (m_down*z(zero_index:end)+(ray_initial_down(i,2)-m_down*ray_initial_down(i,1)))).^2));
[min_dist,index_min_lens_surface] = min(dist);


ray_final_down(i,1) = z(zero_index+index_min_lens_surface);
ray_final_down(i,2) = x(zero_index+index_min_lens_surface);
ray_final_index_down(i,1) = zero_index+index_min_lens_surface;

end

%   Tracing Rays
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Interface 1 : Air-PDMS Interface %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Refraction 
n_left = 1;
n_right = 1.435;
m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    k = ray_final_index_up(i,1);
    m_tangent(i,1) = (x(k+1) - x(k-1))/(z(k+1) - z(k-1));
    phi = abs(atand(m_tangent(i,1)))
    
    
    if(alpha_left_up(i,1)>=0)
        theta_left = abs(90 - alpha_left_up(i,1) - phi)
    elseif(alpha_left_up(i,1)<0)
        theta_left = abs(90 + alpha_left_up(i,1) - phi)
    end
    
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_up(i,1) = -(90 - phi - theta_right);
    
    %below optical axis
    k = ray_final_index_down(i,1);
    m_tangent(i,1) = (x(k+1) - x(k-1))/(z(k+1) - z(k-1));
    phi = abs(atand(m_tangent(i,1)));
    
    if(alpha_left_down(i,1)>=0)
        theta_left = abs(90 - alpha_left_down(i,1) - phi);
    elseif(alpha_left_down(i,1)<0)
        theta_left = abs(90 + alpha_left_down(i,1) - phi);
    end
    
    theta_right = abs(asind((n_left/n_right)*sind(theta_left)));
    
    alpha_right_down(i,1) = -(-90 + phi + theta_right);
    
    

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Interface 2 : PDMS-Glass %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

z_final = z(z_max_index);
m_up = tand(alpha_left_up);
m_down = tand(alpha_left_down);

%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down)

%   Tracing Rays : after refraction
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color)

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);

%   Refraction 
n_left = 1.435;
n_right = 1.52;

m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down(i,1) = theta_right;

end 
 


 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Interface 3 : Glass-Air %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

z_final = z(z_max_index) + t;
m_up = tand(alpha_left_up);
m_down = tand(alpha_left_down);

%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down)

%   Tracing Rays : after refraction
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color)

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);

% Refraction 
n_left = 1.52;
n_right = 1;

m_tangent = zeros(n,1);
alpha_right_up = zeros(n,1);
alpha_right_down = zeros(n,1);
for i = 1:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left));
    
    alpha_right_down(i,1) = theta_right;

end 
 
%%%Right of Interface 2
ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;

z_final = 20000;
m_up = tand(alpha_right_up);
m_down = tand(alpha_right_down);

%   Calculating final co-ordinates of rays
[ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down)

%   Tracing Rays : after refraction
TraceRays_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,color)

%   Angles
[alpha_left_up,alpha_left_down] = AlphaLeft_func(n,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down);

%%%Calculating Focal Length and Longitudinal Spherical Aberration(LSA)

m1 = tand(alpha_left_up);
c1 = ray_initial_up(:,2) - m1.*ray_initial_up(:,1);

m2 = tand(alpha_left_down);
c2 = ray_initial_down(:,2) - m2.*ray_initial_down(:,1);

z_intersection = (c2-c1)./(m1-m2);

Paraxial_focal_length = z_intersection(1,1)
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

z_lcs
min_lcs
beam_t_min

D = 1/(z_lcs*1E-06);
Magnification = (D/4)

figname = 'Magnification';
saveas(f_magnification,strcat(PathName,figname),'fig');


