filename = 'data/singlepositions.csv';
format long;
data = csvread(filename, 1, 0);

time = data(:,1);
pivotL = data(:,2:3);
pivotR = data(:,4:5);
pendulum = data(:,6:7);

% Calculate pendulum length in pixels
l = .342; % pendulum length in m
pivotDiff = mean(pivotR(:,1) - pivotL(:,1));
pixperm = pivotDiff / .206;
lpix = l*pixperm;

% Calculate theta and find zero velocity "starting point"
theta = asin((pendulum(:,1)-pivotR(:,1))/lpix);
dtheta = diff(theta);
posIndex = find(dtheta > 0);
startIndex = posIndex(1);

plot(time(startIndex:end), theta(startIndex:end), time(startIndex:end-1), dtheta(startIndex:end))
