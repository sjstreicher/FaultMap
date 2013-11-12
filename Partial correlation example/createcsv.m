clear all
clc

% Calculate partial correlation matrix for sample data

load('data.mat')

data_mat = data.data;

csvwrite('data.csv', data_mat);

% Need the variables to be according to columns
size(data_mat)



