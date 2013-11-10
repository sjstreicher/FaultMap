clear all
clc

% Generate auto-regressive model as in Shu & Zhao with 5000 points

% x(i) = x(i-1) + y(i-5)

% First get 5100 points of y

y = normrnd(0, 1, [1 5100]);

x = zeros(1, 5000);

for i = 10:5100
    x(i) = x(i-1) + y(i-5);
end

y = y(end-5000+1:end);
x = x(end-5000+1:end);

% Try normalizing x to see what happens

csvwrite('autoregx_data.csv', x');
csvwrite('autoregy_data.csv', y');