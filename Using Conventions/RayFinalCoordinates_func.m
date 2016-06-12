function [ray_final_up,ray_final_down] = RayFinalCoordinates_func(n,z_final,m_up,m_down,ray_initial_up,ray_initial_down)
for i = 1:n
    ray_final_up(i,1) = z_final;
    ray_final_up(i,2) = (m_up(i,1)*ray_final_up(i,1)) + (ray_initial_up(i,2) - m_up(i,1)*ray_initial_up(i,1));
    ray_final_down(i,1) = z_final;
    ray_final_down(i,2) = (m_down(i,1)*ray_final_down(i,1)) + (ray_initial_down(i,2) - m_down(i,1)*ray_initial_down(i,1));
end

end

