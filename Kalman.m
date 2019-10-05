clear all;
close all;

dt = 0.25; % Time step

% Sattelite inertial parameters
m = 10;
w = 0.1;
d = 0.1;
h = 0.3;
Ix = (1/12)*m*(w^2+h^2);
Iy = (1/12)*m*(d^2+h^2);
Iz = (1/12)*m*(w^2+d^2);
% Set I, torque_max, and w0 based on actual sattelite conditions, may also
% have to change N for total time
I = [Ix, 0, 0; 0, Iy, 0; 0, 0, Iz]; % Inertia tensor
torque_max = 0.000005; % Maximum Torque

N = 1200000; % Number of Time Steps

% Initial conditions
w0 = [-7.8,10.6,6.1]; % Initial Angular Velocity
P0 = zeros(7);
x0_hat = ones(7,1);
x0_hat(1) = w0(1);
x0_hat(2) = w0(2);
x0_hat(3) = w0(3);
x0_hat(4) = 0.7;
x0_hat(5) = -0.4;
x0_hat(6) = -0.6;
x0_hat(7) = 0.2;
x0_hat(4:7) = x0_hat(4:7) ./ norm(x0_hat(4:7));

target = [-4;3;-9;-2];
target = target/norm(target);

x_true = x0_hat;
Pk = P0;

u = [0; 0; 0];
M = u;

% Process noise and covariance matrices
eta = 1E-4;
zeta = 1E-4;
omegaDotVar = (eta/2)^2;
qDotVar = (zeta/2)^2;
procVar(1:3) = omegaDotVar .* ones(3,1);
procVar(4:7) = qDotVar .* ones(4,1);
Q0 = diag(procVar);
H = eye(7);

% Measurement noise
delta = 0.5;
gyroVar = (pi*delta/360)^2;

epsilon = 1E-1;
qVar = (epsilon/2)^2;


obsVar(1:3) = gyroVar .* ones(3,1);
obsVar(4:7) = qVar .* ones(4,1);
R = diag(obsVar);

pred = @prediction;
upd = @update;
new_true = @next_true;

% Functions

Om = @(omega) [0, omega(3), -omega(2), omega(1);
               -omega(3), 0, omega(1), omega(2);
               omega(2), -omega(1), 0, omega(3);
               -omega(1), -omega(2), -omega(3), 0];

quaternion_multiply = @(a,b) [a(1)*b(1)-a(2)*b(2)-a(3)*b(3)-a(4)*b(4);
                                a(1)*b(2)+a(2)*b(1)+a(3)*b(4)-a(4)*b(3);
                                a(1)*b(3)-a(2)*b(4)+a(3)*b(2)+a(4)*b(2);
                                a(1)*b(4)+a(2)*b(3)-a(3)*b(2)+a(4)*b(1)];
            

quat_to_euler = @(q) [atan2(2*(q(1)*q(2)+q(3)*q(4)),1-2*(q(2)^2+q(3)^2));
                      asin(2*(q(1)*q(3)-q(4)*q(2)));
                      atan2(2*(q(1)*q(4)+q(2)*q(3)),1-2*(q(3)^2+q(4)^2))];
                  
euler_to_quat = @(e) [cos(e(1)/2)*cos(e(2)/2)*cos(e(3)/2)+sin(e(1)/2)*sin(e(2)/2)*sin(e(3)/2);
                      sin(e(1)/2)*cos(e(2)/2)*cos(e(3)/2)-cos(e(1)/2)*sin(e(2)/2)*sin(e(3)/2);
                      cos(e(1)/2)*sin(e(2)/2)*cos(e(3)/2)+sin(e(1)/2)*cos(e(2)/2)*sin(e(3)/2);
                      cos(e(1)/2)*cos(e(2)/2)*sin(e(3)/2)-sin(e(1)/2)*sin(e(2)/2)*cos(e(3)/2)];

quat_to_matrix = @(q) [1-2*(q(3)^2+q(4)^2), 2*(q(2)*q(3)-q(4)*q(1)), 2*(q(2)*q(4)+q(3)*q(1));
                       2*(q(2)*q(3)+q(4)*q(1)), 1-2*(q(2)^2+q(4)^2), 2*(q(3)*q(4)-q(2)*q(1));
                       2*(q(2)*q(4)-q(3)*q(1)), 2*(q(3)*q(4)+q(2)*q(1)), 1-2*(q(2)^2+q(3)^2)];

 euler_to_matrix = @(e) [1,0,0; 0,cos(e(1)),-sin(e(1)); 0,sin(e(1)),cos(e(1))]*[cos(e(2)),0,sin(e(2)); 0,1,0; -sin(e(2)),0,cos(e(2))]*[cos(e(3)),-sin(e(3)),0; sin(e(3)),cos(e(3)),0; 0,0,1];
 
cal_jaco = @calc_jacobian;

% Result Data Lists
w_list = zeros(N,3);
q_list = zeros(N,4);
x_true_list = zeros(N,7);
z_list = zeros(N,7);
speed_list = zeros(N,1);
true_speed_list = zeros(N,1);
torque_list = zeros(N,1);
err_list = zeros(N,1);
t_list = zeros(N,1);

% Random observation noise
errq = randn*sqrt(qVar);
errw = randn*sqrt(gyroVar);
z = x_true + [randn*sqrt(gyroVar);randn*sqrt(gyroVar);randn*sqrt(gyroVar);randn*sqrt(qVar);randn*sqrt(qVar);randn*sqrt(qVar);randn*sqrt(qVar)]; % Initial measurement
xk = z; % Use first measurement as first data point


% PID parameters

K_p = 0.1;
K_i = 0.4;
K_d = 0.08;
SP = [0.0;0.0;0.0]; % Want 0 angular velocity
q = xk(4:7);
att = quat_to_euler(q);
omega = xk(1:3);
PV = omega;
e_p = SP - PV;
e_n = SP - PV;
a = 0;
In = [0; 0; 0];
                           
for k=1:N    
    % Random process noise
    errqDot = randn*sqrt(qDotVar);
    errwDot = randn*sqrt(omegaDotVar);    
    
            
    % Update true value with random noise
    x_true = new_true(I,dt,x_true,u) + [randn*sqrt(omegaDotVar);randn*sqrt(omegaDotVar);randn*sqrt(omegaDotVar);randn*sqrt(qDotVar);randn*sqrt(qDotVar);randn*sqrt(qDotVar);randn*sqrt(qDotVar)];
    x_true(4:7) = x_true(4:7)/norm(x_true(4:7)); % Normalize
        
    % Kalman Filter
    % Adjust Covariance to prevent numerical instability
    if norm(omega) > 1
        Q = Q0*norm(omega);
    else
        Q = Q0;
    end
    
    F = real(cal_jaco(xk,I,dt)); % Update Jacobian Matrix
    % Random observation noise
    errq = randn*sqrt(qVar);
    errw = randn*sqrt(gyroVar);
    z = x_true + [randn*sqrt(gyroVar);randn*sqrt(gyroVar);randn*sqrt(gyroVar);randn*sqrt(qVar);randn*sqrt(qVar);randn*sqrt(qVar);randn*sqrt(qVar)]; % Measurement
    [xhat_pred, P_pred] = pred(I,dt,Q,xk,u,F,Pk); % Predict next value from previous
    [xk,Pk] = update(H, xhat_pred, P_pred, z, R); % Update using measured value
    xk(4:7) = xk(4:7)/norm(xk(4:7)); % Renormalize quaternion
    
    % PID
    q = xk(4:7);
    att = quat_to_euler(q);
    omega = xk(1:3);
    p = target;
    
    T = dt;
    
    q_1 = q(1);
    q_2 = q(2);
    q_3 = q(3);
    q_4 = q(4);
    
    p_1 = p(1);
    p_2 = p(2);
    p_3 = p(3);
    p_4 = p(4);
    
    w_1 = (-(p_1*q_2 - p_2*q_1 + p_3*q_4 - p_4*q_3)/(T*(p_1*q_1 + p_2*q_2 + p_3*q_3 + p_4*q_4)));
    w_2 = -(p_1*q_3 - p_2*q_4 - p_3*q_1 + p_4*q_2)/(T*(p_1*q_1 + p_2*q_2 + p_3*q_3 + p_4*q_4));
    w_3 = -(p_1*q_4 + p_2*q_3 - p_3*q_2 - p_4*q_1)/(T*(p_1*q_1 + p_2*q_2 + p_3*q_3 + p_4*q_4));
    
    ww = [w_1;w_2;w_3];
    
    qinv = [q(1);-q(2);-q(3);-q(4)];
    wb1 = quat_mult(qinv,quat_mult([0;ww],q));
    wb = wb1(2:4);
    if norm(wb) < 0.01
        SP = wb;
    else
        SP = wb/100;
    end
    PV = omega;
    e_p = e_n; % Error at previous time
    e_n = SP - PV; % Error at current time
    
    In = In+(1/2)*(e_p+e_n)*dt; % Update integral using trapezoidal method
    
    % Remove integral when passing zero error to reduce overshoot
    
    if e_n(1)*In(1) < 0
        In(1) = 0;
    end
    if e_n(2)*In(2) < 0
        In(2) = 0;
    end
    if e_n(3)*In(3) < 0
        In(3) = 0;
    end
        
    a = K_p*e_n + K_i*In + K_d*(e_p-e_n)/dt; % Desired angular acceleration
    u = (I*a + cross(omega,(I*omega))); % Required torque to achieve desired acceleration
    
    if abs(u(1)) > torque_max
        u = u*torque_max/abs(u(1));
    end
    if abs(u(2)) > torque_max
        u = u*torque_max/abs(u(2));
    end
    if abs(u(3)) > torque_max
        u = u*torque_max/abs(u(3));
    end
    % Store results in data lists
    w_list(k,:) = xk(1:3);
    q_list(k,:) = xk(4:7);
    x_true_list(k,:) = x_true;
    z_list(k,:) = z;
    torque_list(k) = norm(u);
    true_speed_list(k) = norm(x_true(1:3));
    speed_list(k) = norm(omega);
    err_list(k) = norm((quat_to_euler(q)-quat_to_euler(target))*180/pi);
    t_list(k) = k*dt;
    
end

% Plot output
plot(t_list,true_speed_list,t_list,speed_list);
title('Speed');
legend('True','Estimate');
figure;
plot(t_list,x_true_list(:,4),t_list,q_list(:,1));
legend('True','Estimate');
title('q1');
figure;
plot(t_list,x_true_list(:,5),t_list,q_list(:,2));
legend('True','Estimate');
title('q2');
figure;
plot(t_list,x_true_list(:,6),t_list,q_list(:,3));
legend('True','Estimate');
title('q3');
figure;
plot(t_list,x_true_list(:,7),t_list,q_list(:,4));
legend('True','Estimate');
title('q4');
figure
plot(t_list,torque_list);
title('Torque');
figure
plot(t_list(end-10000:end),err_list(end-10000:end));
title('Error');


function T3 = quat_tensor(q,T)
qinv = [q(1);-q(2);-q(3);-q(4)];
T2 = [quat_mult(quat_mult(q,[0;T(1:3,1)]),qinv),quat_mult(quat_mult(q,[0;T(1:3,2)]),qinv),quat_mult(quat_mult(q,[0;T(1:3,3)]),qinv)];
T3 = T2(2:4,:);
end

function M = e_to_mat(e)
M = [1,0,0; 0,cos(e(1)),-sin(e(1)); 0,sin(e(1)),cos(e(1))]*[cos(e(2)),0,sin(e(2)); 0,1,0; -sin(e(2)),0,cos(e(2))]*[cos(e(3)),-sin(e(3)),0; sin(e(3)),cos(e(3)),0; 0,0,1];
end

function e = mat_to_e(M)
sy = sqrt(M(1,1)^2 + M(2,1)^2);
singular = sy<1E-6;

if not(singular)
    x = atan2(M(3,2),M(3,3));
    y = atan2(-M(3,1),sy);
    z = atan2(M(2,1),M(1,1));
else
    x = atan2(-M(2,3),M(2,2));
    y = atan2(-M(3,1),sy);
    z = 0
end
e = [x;y;z];
end

function es = all_mat_to_e(M)
if and(not(M(3,1) - 1 < 1E-6),not(M(3,1) + 1 < 1E-6))
    x1 = -arcsin(M(3,1));
    x2 = pi-x1;
    y1 = atan2(M(3,2)/cos(x1),M(3,3)/cos(x1));
    y2 = atan2(M(3,2)/cos(x2),M(3,3)/cos(x2));
    z1 = atan2(M(2,1)/cos(x1),M(1,1)/cos(x1));
    z2 = atan2(M(2,1)/cos(x2),M(1,1)/cos(x1));
    es = [x1,x2;y1,y2;z1,z2];
else
    z = 0;
    if M(3,1)+1 < 1E-6
        x = pi/2;
        y = z + atan2(M(1,2),M(1,3));
    else
        x = -pi/2;
        y = -z + atan2(-M(1,2),-M(1,3));
    end
    es = [x;y;z];
end
end

function q = e_to_quat(e)
q = [cos(e(1)/2)*cos(e(2)/2)*cos(e(3)/2)+sin(e(1)/2)*sin(e(2)/2)*sin(e(3)/2);
    sin(e(1)/2)*cos(e(2)/2)*cos(e(3)/2)-cos(e(1)/2)*sin(e(2)/2)*sin(e(3)/2);
    cos(e(1)/2)*sin(e(2)/2)*cos(e(3)/2)+sin(e(1)/2)*cos(e(2)/2)*sin(e(3)/2);
    cos(e(1)/2)*cos(e(2)/2)*sin(e(3)/2)-sin(e(1)/2)*sin(e(2)/2)*cos(e(3)/2)];
end

function p = scale_q(q,x)
p = [q(1)*x;q(2);q(3);q(4)]/norm([q(1)*x;q(2);q(3);q(4)]);
end

function e = quat_to_e(q)
e = [atan2(2*(q(1)*q(2)+q(3)*q(4)),1-2*(q(2)^2+q(3)^2));
    asin(2*(q(1)*q(3)-q(4)*q(2)));
    atan2(2*(q(1)*q(4)+q(2)*q(3)),1-2*(q(3)^2+q(4)^2))];
end

function q = quat_mult(a,b)
q = [a(1)*b(1)-a(2)*b(2)-a(3)*b(3)-a(4)*b(4);
    a(1)*b(2)+a(2)*b(1)+a(3)*b(4)-a(4)*b(3);
    a(1)*b(3)-a(2)*b(4)+a(3)*b(1)+a(4)*b(2);
    a(1)*b(4)+a(2)*b(3)-a(3)*b(2)+a(4)*b(1)];
end


% Calculate next true value of x
function x = next_true(I,T,x_prev,u_prev)
w_prev = x_prev(1:3);
q_prev = x_prev(4:7);
qinv = [q_prev(1);-q_prev(2);-q_prev(3);-q_prev(4)];
ww1 = quat_mult(q_prev,quat_mult([0;w_prev],qinv));
ww = ww1(2:4);
Omega = [0, -ww(1), -ww(2), -ww(3);
         ww(1), 0, -ww(3), ww(2);
         ww(2), ww(3), 0, -ww(1);
         ww(3), -ww(2), ww(1), 0];
c1 = x_prev(3);
c2 = x_prev(1);
c3 = x_prev(2);
a = (I(2,2)-I(3,3))/I(1,1);
b = (I(3,3)-I(1,1))/I(2,2);
wNext = real([(1/2)*c2*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)+1) + c3*sqrt(a)*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)-1)/(2*sqrt(b)) + u_prev(1)*T/I(1,1); (1/2)*c3*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)+1) + c2*sqrt(b)*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)-1)/(2*sqrt(a)) + u_prev(2)*T/I(2,2); c1+u_prev(3)*T/I(3,3)]);
qNext = expm(0.5 * Omega * T) * q_prev;
x = [wNext; qNext];
end

% Predict next x value from previous
function [xhat_pred, P_pred] = prediction(I,T,Q,xhat_prev,u_prev,F,P)
what_prev = xhat_prev(1:3);
qhat_prev = xhat_prev(4:7);
qinv = [qhat_prev(1);-qhat_prev(2);-qhat_prev(3);-qhat_prev(4)];
ww1 = quat_mult(qhat_prev,quat_mult([0;what_prev],qinv));
ww = ww1(2:4);
Omega = [0, -ww(1), -ww(2), -ww(3);
         ww(1), 0, -ww(3), ww(2);
         ww(2), ww(3), 0, -ww(1);
         ww(3), -ww(2), ww(1), 0];
     
c1 = xhat_prev(3);
c2 = xhat_prev(1);
c3 = xhat_prev(2);
a = (I(2,2)-I(3,3))/I(1,1);
b = (I(3,3)-I(1,1))/I(2,2);
wNext = real([(1/2)*c2*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)+1) + c3*sqrt(a)*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)-1)/(2*sqrt(b)) + u_prev(1)*T/I(1,1); (1/2)*c3*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)+1) + c2*sqrt(b)*exp(-sqrt(a*b)*c1*T)*(exp(2*sqrt(a*b)*c1*T)-1)/(2*sqrt(a)) + u_prev(2)*T/I(2,2); c1+u_prev(3)*T/I(3,3)]);
qNext = expm(0.5 * Omega * T) * qhat_prev;
xhat_pred = [wNext; qNext];
P_pred = F * P * F' + Q;
end

% Update x value using predicted and measured
function [xk, Pk] = update(H, xhat_pred, P_pred, z, R)
y = z - xhat_pred;
S = H * P_pred * H' + R;
K = P_pred * H * S^-1;
xk = xhat_pred + K*y;
Pk = (eye(7) - K*H) * P_pred;
end

% Calculate the jacobian matrix
function j = calc_jacobian(x,I,T)
w1 = x(1);
w2 = x(2);
w3 = x(3);
q1 = x(4);
q2 = x(5);
q3 = x(6);
q4 = x(7);
I1_1 = I(1,1);
I1_2 = I(1,2);
I1_3 = I(1,3);
I2_1 = I(2,1);
I2_2 = I(2,2);
I2_3 = I(2,3);
I3_1 = I(3,1);
I3_2 = I(3,2);
I3_3 = I(3,3);

j = [                                                                                                                                                                                                                                                                                                                                                                                                                           1 - T*(((I1_2*I2_3 - I1_3*I2_2)*(2*I2_1*w1 - I1_1*w2 + I2_2*w2 + I2_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) - ((I2_2*I3_3 - I2_3*I3_2)*(I2_1*w3 - I3_1*w2))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_2*I3_3 - I1_3*I3_2)*(2*I3_1*w1 - I1_1*w3 + I3_2*w2 + I3_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                               T*(((I1_2*I3_3 - I1_3*I3_2)*(I1_2*w3 - I3_2*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_2*I2_3 - I1_3*I2_2)*(I1_1*w1 + 2*I1_2*w2 + I1_3*w3 - I2_2*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) - ((I2_2*I3_3 - I2_3*I3_2)*(I3_1*w1 - I2_2*w3 + 2*I3_2*w2 + I3_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                               T*(((I1_2*I2_3 - I1_3*I2_2)*(I1_3*w2 - I2_3*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_2*I3_3 - I1_3*I3_2)*(I1_1*w1 + I1_2*w2 + 2*I1_3*w3 - I3_3*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I2_2*I3_3 - I2_3*I3_2)*(I2_1*w1 + I2_2*w2 + 2*I2_3*w3 - I3_3*w2))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0;
                                                                                                                                                                                                                                                                                                                                                                                                                               T*(((I1_1*I2_3 - I1_3*I2_1)*(2*I2_1*w1 - I1_1*w2 + I2_2*w2 + I2_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) - ((I2_1*I3_3 - I2_3*I3_1)*(I2_1*w3 - I3_1*w2))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_1*I3_3 - I1_3*I3_1)*(2*I3_1*w1 - I1_1*w3 + I3_2*w2 + I3_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                           1 - T*(((I1_1*I3_3 - I1_3*I3_1)*(I1_2*w3 - I3_2*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_1*I2_3 - I1_3*I2_1)*(I1_1*w1 + 2*I1_2*w2 + I1_3*w3 - I2_2*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) - ((I2_1*I3_3 - I2_3*I3_1)*(I3_1*w1 - I2_2*w3 + 2*I3_2*w2 + I3_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                              -T*(((I1_1*I2_3 - I1_3*I2_1)*(I1_3*w2 - I2_3*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_1*I3_3 - I1_3*I3_1)*(I1_1*w1 + I1_2*w2 + 2*I1_3*w3 - I3_3*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I2_1*I3_3 - I2_3*I3_1)*(I2_1*w1 + I2_2*w2 + 2*I2_3*w3 - I3_3*w2))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0;
                                                                                                                                                                                                                                                                                                                                                                                                                              -T*(((I1_1*I2_2 - I1_2*I2_1)*(2*I2_1*w1 - I1_1*w2 + I2_2*w2 + I2_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) - ((I2_1*I3_2 - I2_2*I3_1)*(I2_1*w3 - I3_1*w2))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_1*I3_2 - I1_2*I3_1)*(2*I3_1*w1 - I1_1*w3 + I3_2*w2 + I3_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                               T*(((I1_1*I3_2 - I1_2*I3_1)*(I1_2*w3 - I3_2*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_1*I2_2 - I1_2*I2_1)*(I1_1*w1 + 2*I1_2*w2 + I1_3*w3 - I2_2*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) - ((I2_1*I3_2 - I2_2*I3_1)*(I3_1*w1 - I2_2*w3 + 2*I3_2*w2 + I3_3*w3))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)),                                                                                                                                                                                                                                                                                                                                                                                                                           T*(((I1_1*I2_2 - I1_2*I2_1)*(I1_3*w2 - I2_3*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I1_1*I3_2 - I1_2*I3_1)*(I1_1*w1 + I1_2*w2 + 2*I1_3*w3 - I3_3*w1))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1) + ((I2_1*I3_2 - I2_2*I3_1)*(I2_1*w1 + I2_2*w2 + 2*I2_3*w3 - I3_3*w2))/(I1_1*I2_2*I3_3 - I1_1*I2_3*I3_2 - I1_2*I2_1*I3_3 + I1_2*I2_3*I3_1 + I1_3*I2_1*I3_2 - I1_3*I2_2*I3_1)) + 1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0;
 - (T*q2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(T*q1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*q4*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - (T*q3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(T*q1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q4*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - (T*q4)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(T*q1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q4*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),          1/(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2) - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*w1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*w3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*w1*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) - T*w3*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)) - (T*w1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)), - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)) - (T*w2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)), - ((q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*(2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)) - (T*w3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2));
   (T*q1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(T*q1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*q4*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - (T*q4)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(T*q1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q4*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*q3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(T*q1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q4*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), (T*w1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*w1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*w3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),            1/(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2) - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*w1*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) - T*w3*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*w3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - ((q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*(2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)) - (T*w2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2));
   (T*q4)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(T*q1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*q4*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*q1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(T*q1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q4*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - (T*q2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(T*q1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q4*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), (T*w2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*w1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*w3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*w1*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) - T*w3*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)) - (T*w3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)),            1/(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2) - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*w1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*(2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2));
 - (T*q3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(T*q1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*q4*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*q2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(T*q1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q4*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*q1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(T*q1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*q2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*q3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*q4*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), (T*w3)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) + T*w1*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) + T*w3*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)),   (T*w2)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)) - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) + T*w2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) - T*w1*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2)) - T*w3*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)), - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(2*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w1*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w3*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w2*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2)) - (T*w1)/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2)),            1/(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(1/2) - ((q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*(2*abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))*sign(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2)) + T*w1*abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))*sign(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2)) - T*w2*abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))*sign(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2)) - T*w3*abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))*sign(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))))/(2*(abs(q2 + T*((q1*w1)/2 + (q3*w3)/2 - (q4*w2)/2))^2 + abs(q3 + T*((q1*w2)/2 - (q2*w3)/2 + (q4*w1)/2))^2 + abs(q4 + T*((q1*w3)/2 + (q2*w2)/2 - (q3*w1)/2))^2 + abs(q1 - T*((q2*w1)/2 + (q3*w2)/2 + (q4*w3)/2))^2)^(3/2))];

end
