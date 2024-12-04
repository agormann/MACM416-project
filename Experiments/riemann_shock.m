clear;
clc;
close all;
%% Setup
nx = 400;
dx = 4/nx;
X = dx*(-nx/2:nx/2-1);

uL = 1;
uR = 0;
u0 = uL*(X<0) + uR*(X>0);
%% Solving
[T,U] = godunov_dirichlet(X,u0,2); % numerical solution
R = riemann(uL,uR,X,T); % exact solution
%% Plotting
padding = 0.10*(max(u0)-min(u0));

f = figure();
f.Position = [100 100 480 360];
v = VideoWriter([pwd '/Movies/riemann_shock.mp4'], 'MPEG-4');
v.FrameRate = 30;
open(v);

for i = 1:length(T)
    % error
    absErr = norm(R(i,:) - U(i,:));
    relErr = norm((R(i,:) - U(i,:)) / U(i,:));

    cla; % clear frame
    plot(X, R(i,:), 'k-', 'LineWidth', 2, 'DisplayName', "Exact solution")
    hold on;
    plot(X, U(i,:), 'r:', 'LineWidth', 2, 'DisplayName', "Godunov method");

    % plot settings
    title(sprintf('t = %.3f, Error_{abs} = %.3f, Error_{rel} = %.3f', T(i), absErr, relErr));
    grid on;
    xlabel('x');
    xlim([min(X),max(X)]);
    ylabel('u(x,t)');
    ylim([min(u0)-padding,max(u0)+padding]);

    % Write current frame to video
    frame = getframe(f);
    writeVideo(v, frame);
    pause(0.01);
end

close(v);

disp('Animation saved as riemann_shock.mp4');