clear all
clc

%% Export cvs files for use in Python
% Load variables

load('original');
original_data = original.data;

load('puredelay');
puredelay_data = puredelay.data;

load('delayedtf');
load('doubledelayedtf');

csvwrite('original_data.csv', original_data(5002:end));
csvwrite('puredelay_data.csv', puredelay_data(5002:end));
csvwrite('delayedf_data.csv', delayedtf.data(5002:end));
csvwrite('doubledelayedf_data.csv', doubledelayedtf.data(5002:end));


