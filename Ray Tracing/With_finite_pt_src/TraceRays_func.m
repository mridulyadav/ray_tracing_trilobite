function [] = TraceRays_func(n1,n2,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down,c)
for i = n1:n2
    Ray_Z_up = [ray_initial_up(i,1) ray_final_up(i,1)];
    Ray_X_up = [ray_initial_up(i,2) ray_final_up(i,2)];
    line(Ray_Z_up,Ray_X_up,'Color',c)
    
    Ray_Z_down = [ray_initial_down(i,1) ray_final_down(i,1)];
    Ray_X_down = [ray_initial_down(i,2) ray_final_down(i,2)];
    line(Ray_Z_down,Ray_X_down,'Color',c)
end
end

