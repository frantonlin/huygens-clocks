function non_lin()
d_cart = 0.0277/9.8; %friction of the cart(acceloration) in m/s^2 as measured by tracker
d_bob = 0.01; %damping rate of the bob traveling through the air(viscous)

m_bob = 67.33;%mass of the just the bob
m_total = 2.13; %mass of the total system

mu = m_bob/m_total;
g = 0; %little gamma
G = 0; %big gamma
gravity = 9.8; %m/s2
l_pendulum = 0.342; %m
time_dimensional = sqrt((gravity/l_pendulum)); %simensional time contant to adjust model to correct period

angle_start = deg2rad(15); %starting position of bob
x0 = [angle_start; 0];
t0 = [0 1000];

options = odeset('RelTol', 1e-8);
[T,X] = ode45(@derivs, t0, x0, options);
angle = rad2deg(X(:,1));
time_adjust = T/time_dimensional; %corrects time into a dimensional space

plot(time_adjust, angle, 'b')
xlabel('Time (s)')
ylabel('Degrees')


function dxdt = derivs(t, x)
    theta = x(1);
    w = x(2);
    
    d_theta_dt = w;
    d_w_dt = -sin(theta) - 2*d_bob*w*abs(w);% - vd_bob*sign(w); %driving, damping, viscous friction
    
    if sign(w)<0 && theta >= deg2rad(.75) && theta >= deg2rad(-.75)
        d_w_dt = d_w_dt - .002;
    end
    
    dxdt= [d_theta_dt;d_w_dt];
    
end

x0 = [omega_1; w_1; omega_2; w_2; y; v]; %initial conditions for multi-pendulum

function res = multiclock(t, x)
    A = [0 1 0 0 0 0;
         -1 -2*g 0 0 0 0;
         0 0 0 1 0 0;
         0 0 -1 -2*g 0 0;
         0 0 0 0 0 1;
         0 0 0 0 0 -2*G];
     
    B = [1 0 0 0 0 0;
         0 1 0 0 0 1;
         0 0 1 0 0 0;
         0 0 0 1 0 1;
         0 0 0 0 1 0;
         0 mu 0 mu 0 1];

    u = [omega_prime_1; w_prime_1;omega_prime_2; w_prime_2; y_prime; v_prime];

    res = A*x + B*u;
end

end