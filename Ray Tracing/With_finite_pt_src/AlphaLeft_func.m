function [alpha_left_up,alpha_left_down] = AlphaLeft_func(n1,n2,ray_initial_up,ray_initial_down,ray_final_up,ray_final_down)

 for i = n1:n2
    alpha_left_up(i,1) = atand((ray_final_up(i,2)-ray_initial_up(i,2))/(ray_final_up(i,1)-ray_initial_up(i,1)));
    alpha_left_down(i,1) = atand((ray_final_down(i,2)-ray_initial_down(i,2))/(ray_final_down(i,1)-ray_initial_down(i,1)));
 end

end

