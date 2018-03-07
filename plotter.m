filename = 'antiphasepositions.csv';
format long;
M = csvread(filename, 1, 0);

steps = size(M,1);

time = M(:,1);
thetaLeft = zeros(1, steps);
xLeft = M(:,6);
pivotxLeft = M(:,2);
posLeft = xLeft - pivotxLeft;

thetaRight = zeros(1, steps);
xRight = M(:,8);
pivotxRight = M(:,4);
posRight = xRight - pivotxRight;

rmeters = .342; % pendulum length in m
pivotDiff = mean(pivotxRight - pivotxLeft)
pixperm = pivotDiff / .206;
rpix = rmeters*pixperm;

for i = 1:steps
    thetaLeft(i) = asin(posLeft(i)/rpix);
    thetaRight(i) = asin(posRight(i)/rpix);
end

hold on
plot(time, thetaLeft, 'b') 
plot(time, thetaRight, 'r')