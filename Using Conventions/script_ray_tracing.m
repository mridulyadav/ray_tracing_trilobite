




X; % column 1 : x-axis ; column 2 : z-axis(sag) with x = 0 at z = 0
R; %Radius of curvature at the vertex
z = -X(:,2);% -ve curvature
x = X(:,1);

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


%%%Drawing Optical Axis/Co-ordinate system
%Z-Axis 
z_optical_axis_min = -10000;
z_optical_axis_max = 10000;
Z_optical_axis_z = [z_optical_axis_min z_optical_axis_max];
Z_optical_axis_x = [0 0];
line(Z_optical_axis_z,Z_optical_axis_x)

%X-Axis 
x_optical_axis_min = -5000;
x_optical_axis_max = 5000;
X_optical_axis_z = [0 0];
X_optical_axis_x = [x_optical_axis_min x_optical_axis_max];
line(X_optical_axis_z,X_optical_axis_x)

%%%Drawing the lens
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
paraxial_angle = 5;%in degrees
min_height = R*tand(paraxial_angle);

diff = abs(min_height - x(1:zero_index,1));
[min_diff,index_min_height] = min(diff);


if(index_min_height<zero_index)
    min_height = abs(x(zero_index_1+10));
end

c1 = abs(X(1,1));
c2 = abs(X(end,1));

c3 = min(c1,c2);%maximum height i.e. height of final ray; this height is divided into n equal parts
max_height = c3-min_height;

%2 : x
%1 : z
%%%Point Source at finite distance
%Location of point source
z_point_source = -150000;
x_point_source = 0;
for i = 1:n
    ray_initial_up(i,1) = z_point_source;
    ray_initial_up(i,2) = x_point_source;
    ray_initial_down(i,1) = z_point_source;
    ray_initial_down(i,2) = x_point_source;
end

% %Point Source at infinity
% %Ray origin for display
% z_ray_origin = -5000;
% for i = 1:n
%     ray_initial_up(i,1) = z_ray_origin;
%     ray_initial_up(i,2) = min_height + (max_height/n)*(i-1);
%     ray_initial_down(i,1) = z_ray_origin;
%     ray_initial_down(i,2) = -min_height-(max_height/n)*(i-1);
% end


    
    
    
    
  
for i = 1:n
    ray_i_x_up = min_height + (max_height/n)*(i-1);
    
    diff = abs(ray_i_x_up - x(1:zero_index,1));
    [min_diff,index_ray_i_x] = min(diff);
    
    ray_final_up(i,1) = z(index_ray_i_x,1);
    ray_final_up(i,2) = x(index_ray_i_x,1);
    ray_final_index_up(i,1) = index_ray_i_x;

    diff = [];
    ray_i_x_down = -min_height-(max_height/n)*(i-1)
    diff = abs(ray_i_x_down - x(zero_index:end,1))
    [min_diff,index_ray_i_x] = min(diff)
    
    ray_final_down(i,1) = z(zero_index+index_ray_i_x,1);
    ray_final_down(i,2) = x(zero_index+index_ray_i_x,1);
    ray_final_index_down(i,1) = zero_index+index_ray_i_x;
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

n_left = 1.4;
n_right = 1; 
 
%%% Checking for maximum angle before TIR
flag_TIR = 0;
m_tangent = [];

for i = 1:n
    %above optical axis
    k = ray_final_index_up(i,1);
    m_tangent(i,1) = (x(k+1) - x(k-1))/(z(k+1) - z(k-1));
    phi = abs(atand(m_tangent(i,1)));
    
    if(alpha_left_up(i,1)>0)
        theta_left = abs(90 - alpha_left_up(i,1) - phi);
    elseif(alpha_left_up(i,1)<0)
        theta_left = abs(90 + alpha_left_up(i,1) - phi);
    end
    
    if(sind(theta_left)>= (n_right/n_left))
        flag_TIR = 1;
        k_max_TIR_up = k;
    end
    
    %below optical axis
    k = ray_final_index_down(i,1);
    m_tangent(i,1) = (x(k+1) - x(k-1))/(z(k+1) - z(k-1));
    phi = abs(atand(m_tangent(i,1)));
    
    if(alpha_left_down(i,1)>0)
        theta_left = abs(90 - alpha_left_down(i,1) - phi)
    elseif(alpha_left_down(i,1)<0)
        theta_left = abs(90 + alpha_left_down(i,1) -phi)
    end
    
    if(sind(theta_left)>= (n_right/n_left))
        flag_TIR = 1;
        k_max_TIR_down = k;
    end
    
end

flag_TIR
if(flag_TIR == 1)
    max_height = (min(abs(x(k_max_TIR_up)),abs(x(k_max_TIR_down))))-1.5*min_height;
end

% %%Point Source at infinity
% %Ray origin for display
% z_ray_origin = -5000;
% for i = 1:n
%     ray_initial_up(i,1) = z_ray_origin;
%     ray_initial_up(i,2) = min_height + (max_height/n)*(i-1);
%     ray_initial_down(i,1) = z_ray_origin;
%     ray_initial_down(i,2) = -min_height-(max_height/n)*(i-1);
% end


%%% Interface 1 : Air-Glass Interface

t = 1;%microns : Thickness of glass plate
%Drawing the interface
[z_min,z_min_index] = min(z);

Z_air_glass_Int = [(z(z_min_index)-t) (z(z_min_index)-t)];

if(z_min_index<zero_index)
    X_air_glass_Int = [x(z_min_index) x(end)];
elseif(z_min_index>zero_index)
    X_air_glass_Int = [x(z_min_index) x(1)];
end
    
line(Z_air_glass_Int,X_air_glass_Int,'Color','k','LineWidth',1.5);

for i = 1:n
    ray_final_up(i,1) = z(z_min_index) - t;
    ray_final_up(i,2) = min_height + (max_height/n)*(i-1);
    ray_final_down(i,1) = z(z_min_index) - t;
    ray_final_down(i,2) = -min_height - (max_height/n)*(i-1);
end
    
% Angles
alpha_left_up = [];
alpha_left_down = [];
 for i = 1:n
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)));
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)));
 end


 %Tracing Rays : before refraction
for i = 1:n
    Ray_Z_up = [ray_initial_up(i,1) ray_final_up(i,1)];
    Ray_X_up = [ray_initial_up(i,2) ray_final_up(i,2)];
    line(Ray_Z_up,Ray_X_up)
    
    Ray_Z_down = [ray_initial_down(i,1) ray_final_down(i,1)];
    Ray_X_down = [ray_initial_down(i,2) ray_final_down(i,2)];
    line(Ray_Z_down,Ray_X_down)
end


% Refraction 
n_left = 1;
n_right = 1.52;

m_tangent = [];
alpha_right_up = [];
alpha_right_down = [];
for i = 1:n
    %above optical axis
    theta_left = alpha_left_up(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_up(i,1) = theta_right;
    
    %below optical axis
    theta_left = alpha_left_down(i,1);
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_down(i,1) = theta_right;

end


%%%Interface 2 : Glass-Pdms

ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

for i = 1:n
%above optical axis
m_up = tand(alpha_left_up(i,1));
ray_final_up(i,1) = z(z_min_index);
ray_final_up(i,2) = m_up*ray_final_up(i,1) + (ray_initial_up(i,2) - m_up*ray_initial_up(i,1));

%below optical axis
m_down = tand(alpha_left_down(i,1));
ray_final_down(i,1) = z(z_min_index);
ray_final_down(i,2) = m_down*ray_final_down(i,1) + (ray_initial_down(i,2) - m_down*ray_initial_down(i,1));

end

% Angles
alpha_left_up = [];
alpha_left_down = [];
 for i = 1:n
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)));
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)));
 end


%Tracing Rays : before refraction
for i = 1:n
    Ray_Z_up = [ray_initial_up(i,1) ray_final_up(i,1)];
    Ray_X_up = [ray_initial_up(i,2) ray_final_up(i,2)];
    line(Ray_Z_up,Ray_X_up)
    
    Ray_Z_down = [ray_initial_down(i,1) ray_final_down(i,1)];
    Ray_X_down = [ray_initial_down(i,2) ray_final_down(i,2)];
    line(Ray_Z_down,Ray_X_down)
end
    
% Refraction 
n_left = 1.52;
n_right = 1.4;

m_tangent = [];
alpha_right_up = [];
alpha_right_down = [];
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
    
%%% Interface 3
ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;


ray_final_index_up = [];
ray_final_index_down = [];

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
%Tracing Rays : before refraction
for i = 1:n
    Ray_Z_up = [ray_initial_up(i,1) ray_final_up(i,1)];
    Ray_X_up = [ray_initial_up(i,2) ray_final_up(i,2)];
    line(Ray_Z_up,Ray_X_up)
    
    Ray_Z_down = [ray_initial_down(i,1) ray_final_down(i,1)];
    Ray_X_down = [ray_initial_down(i,2) ray_final_down(i,2)];
    line(Ray_Z_down,Ray_X_down)
end

% Angles
alpha_left_up = [];
alpha_left_down = [];
 for i = 1:n
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)))
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)))
 end



%%%Refraction 
n_left = 1.4;
n_right = 1;
m_tangent = [];
alpha_right_up = [];
alpha_right_down = [];
for i = 1:n
    %above optical axis
    k = ray_final_index_up(i,1);
    m_tangent(i,1) = (x(k+1) - x(k-1))/(z(k+1) - z(k-1));
    phi = abs(atand(m_tangent(i,1)))
    
    
    if(alpha_left_up(i,1)>0)
        theta_left = abs(90 - alpha_left_up(i,1) - phi)
    elseif(alpha_left_up(i,1)<0)
        theta_left = abs(90 + alpha_left_up(i,1) - phi)
    end
    
    theta_right = asind((n_left/n_right)*sind(theta_left))
    
    alpha_right_up(i,1) = 90 - phi - theta_right;
    
    %below optical axis
    k = ray_final_index_down(i,1);
    m_tangent(i,1) = (x(k+1) - x(k-1))/(z(k+1) - z(k-1));
    phi = abs(atand(m_tangent(i,1)));
    
    if(alpha_left_down(i,1)>0)
        theta_left = abs(90 - alpha_left_down(i,1) - phi);
    elseif(alpha_left_down(i,1)<0)
        theta_left = abs(90 + alpha_left_down(i,1) - phi);
    end
    
    theta_right = abs(asind((n_left/n_right)*sind(theta_left)));
    
    alpha_right_down(i,1) = -90 + phi + theta_right
    
    

end




%%%Right of Interface 1
ray_initial_up = ray_final_up;
ray_initial_down = ray_final_down;
alpha_left_up = alpha_right_up;
alpha_left_down = alpha_right_down;

z_final = 20000;


for i = 1:n
    ray_final_up(i,1) = z_final;        
    m = tand(alpha_right_up(i,1));
    ray_final_up(i,2) = m*ray_final_up(i,1) + (ray_initial_up(i,2) - m*ray_initial_up(i,1));
    
    ray_final_down(i,1) = z_final;
    m = tand(alpha_right_down(i,1));
    ray_final_down(i,2) = m*ray_final_down(i,1) + (ray_initial_down(i,2) - m*ray_initial_down(i,1));
    
end

%Tracing Rays : after refraction
for i = 1:n
    Ray_Z_up = [ray_initial_up(i,1) ray_final_up(i,1)];
    Ray_X_up = [ray_initial_up(i,2) ray_final_up(i,2)];
    line(Ray_Z_up,Ray_X_up)
    
    Ray_Z_down = [ray_initial_down(i,1) ray_final_down(i,1)];
    Ray_X_down = [ray_initial_down(i,2) ray_final_down(i,2)];
    line(Ray_Z_down,Ray_X_down)
end

% Angles
alpha_left_up = [];
alpha_left_down = [];
 for i = 1:n
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)))
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)))
 end




%%%Calculating Focal Length and Longitudinal Spherical Aberration(LSA)

m1 = tand(alpha_left_up);
c1 = ray_initial_up(:,2) - m1.*ray_initial_up(:,1);

m2 = tand(alpha_left_down);
c2 = ray_initial_down(:,2) - m2.*ray_initial_down(:,1);

z_intersection = (c2-c1)./(m1-m2);

Paraxial_focal_length = z_intersection(1,1)
Meridinal_focal_length = z_intersection(n,1)

LSA = Paraxial_focal_length - Meridinal_focal_length




%%%Calculating Surface of Least Confusion
m_slope_up = [];
m_slope_down = [];
min_lcs = Inf;
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
    
