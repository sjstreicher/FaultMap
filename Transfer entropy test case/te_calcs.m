clear all
clc

% Calculate modified tranfer entropy between various signals

% Load variables

load('original');
original_data = original.data;

load('puredelay');
puredelay_data = puredelay.data;

% Get PDF for data

[f_original, xi_original] = ksdensity(original_data);
[f_delay, xi_delay] = ksdensity(puredelay_data);

% Calculate own PDF using Gaussian Kernel function

% Discrete implementation (Bauer, 2005:135)

% Generate discrete grid

x_max = max(original_data);
x_min = min(original_data);
x_n = length(original_data);
n_amp = 1000; % Number of amplitude bins
x_discrete = zeros(1, x_n);
for i = 1:x_n
    x_discrete(i) = round((n_amp - 1) * ((original_data(i) - x_min) / (x_max - x_min))) + 1;
end


% Calculate discrete Kernel width 

% Calculate scaling factor a




% Use 100 equally spaced points in the range of samples


% 
% x_space = linspace(x_min, x_max, 100);








