clear;
clc;
close all;
%% setup
nx = 100;
dx = 2*pi/nx;
X = dx*(-nx/2:nx/2-1);

u0 = (X>0.5*pi) & (X<1.5*pi); % initial profile

[T,U] = godunov(X,u0,20); % solution
%% making movie
padding = 0.10*(max(u0)-min(u0));

% Set up figure and video writer
f = figure();
f.Position = [100,100,640,640];
v = VideoWriter('symmetric_step.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

for i = 1:length(T)
    plot(X, U(i,:), '-', 'LineWidth', 2);
    title(sprintf('Time: %.3f', T(i)));
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

disp('Animation saved as animation.mp4');