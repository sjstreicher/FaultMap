clear all
clc

% Calculate modified tranfer entropy between various signals

% Load variables

load('original');
original_data = original.data;

load('puredelay');
puredelay_data = puredelay.data;

c = (4/3)^(1/5); % Constant given in text

% Get PDF for data

[f_original, xi_original] = ksdensity(original_data);
[f_delay, xi_delay] = ksdensity(puredelay_data);


%% Single variable PDF calculation
% Calculate own PDF using Gaussian Kernel function

% Discrete implementation (Bauer, 2005:135) - no longer used
% Using form given by Shu & Zhao (2013:174)

x_n = length(original_data);
x_max = max(original_data);
x_min = min(original_data);

% Calculate standard deviation for each data set

sig_original = std(original_data);
sig_puredelay = std(original_data);

% Calculate theta for single variable PDF

theta_original = c * sig_original * x_n^(-1/5);



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

%% Multivariable PDF calculation

% Need to divide range into discrete points
n_amp = 100; % Number of amplitude bins

% Store data in single variable

data = [original_data, puredelay_data];

% Specify number of varibles
n_var = length(data(1, :));

% Store length of each variable
x_n = length(data(:, 1));

% Get maximum and minimum of each variable and generate space
% x_max = zeros(n_var, 1);
% x_min = zeros(n_var, 1);
% x_space = zeros(n_var, n_amp);
% for v = 1:n_var
%     x_max(v, 1) = max(data(:, v));
%     x_min(v, 1) = min(data(:, v));
%     x_space(v, :) = linspace(x_min(v), x_max(v), n_amp);
% end

% Calculate theta for each data set
% theta = zeros(n_var, 1);
% for s = 1:n_var
%     theta(s) = c * std(data(:, s)) * x_n^(-1/(4+s));
% end

% p = zeros(n_var, n_amp);
% K_prod = 0;
% K_sum = 0;

% for nv = 1:n_var
%  
% 
% 
%     for k = 1:n_amp   
%         for i = 1:x_n
%             
%             % Calculate product of kernel functions at a specific bin
%             % reference
%             temp_prod = 1;
%             for p = 1:n_var
%                 temp_kernel = single_kernel(x_space(p, k), data(i, p), theta(p));
%                 temp_prod = temp_prod * temp_kernel;
%             end
%             
%             temp_sum = single_kernel(x_space(k), original_data(i), theta_original);
%             K_sum = K_sum + temp_sum;
%         end
%         p(k, k)
%         K_sum = 0;
%     end
%     p(k) = (1/x_n) * K_sum;
% end











