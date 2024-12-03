clear;
clc;
close all;
%% Setup
nx = 1000;
dx = 2*pi/nx;
X = dx*(1:nx);
u0 = sin(X).^2; % sine-squared initial profile
%% Solving
[T,U] = godunov(X,u0,10); % numerical solution
%% Plotting
padding = 0.10*(max(u0)-min(u0));

f = figure();
f.Position = [100,100,640,640];
v = VideoWriter([pwd '/Movies/sine_squared_wave.mp4'], 'MPEG-4');
v.FrameRate = 60;
open(v);

for i = 1:length(T)
    plot(X, U(i,:), 'r:', 'LineWidth', 2);

    % plot settings
    title(sprintf('t = %.3f', T(i)));
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

disp('Animation saved as sine_squared_wave.mp4');