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

x_n = length(original_data);
x_max = max(original_data);
x_min = min(original_data);

% Calculate own PDF using Gaussian Kernel function

% Discrete implementation (Bauer, 2005:135)
% Using form given by Shu & Zhao (2013:174)

% Calculate standard deviation for each data set

sig_original = std(original_data);
sig_puredelay = std(original_data);

% Calculate theta for single variable PDF
c = (4/3)^(1/5); % Constant given in text
theta_original = c * sig_original * x_n^(-1/5);

%% Single variable PDF calculation

% Need to divide range into discrete points
n_amp = 100; % Number of amplitude bins

x_space = linspace(x_min, x_max, n_amp);

p = zeros(1, n_amp);
K_sum = 0;
for k = 1:n_amp   
    for i = 1:x_n
        temp_sum = single_kernel(x_space(k), original_data(i), theta_original);
        K_sum = K_sum + temp_sum;
    end
    p(k) = (1/x_n) * K_sum;
    K_sum = 0;
end

% Generate discrete grid

% x_discrete = zeros(1, x_n);
% for i = 1:x_n
%     x_discrete(i) = round((n_amp - 1) * ((original_data(i) - x_min) / (x_max - x_min))) + 1;
% end





