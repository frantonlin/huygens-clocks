filename = 'singlepositions.csv';
format long;
M = csvread(filename, 1, 0);

steps = size(M,1);

time = M(:,1);
thetaBlue = zeros(1, steps);
xBlue = M(:,6);
yBlue = M(:,7);

for i = 1:steps
    x = xBlue(i);
    y = yBlue(i);
    thetaBlue(i) = atan(y / x);
end


plot(time, thetaBlue) 