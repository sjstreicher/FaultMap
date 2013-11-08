function K = single_kernel(x, xs, theta)
    K = (1/((sqrt(2*pi))*theta))*exp(-(x - xs)^2 / (2*(theta^2)));
end








