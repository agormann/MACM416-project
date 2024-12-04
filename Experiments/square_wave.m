clear;
clc;
close all;
%% Setup
nx = 1000;
dx = 2*pi/nx;
X = dx*(1:nx);
u0 = (X>0.5*pi) & (X<1.5*pi); % square initial profile
%% Solving
[T,U] = godunov(X,u0,10); % numerical solution
%% Movie plot
padding = 0.10*(max(u0)-min(u0));

f = figure();
f.Position = [100,100,360,240];
%v = VideoWriter([pwd '/Movies/square_wave.mp4'], 'MPEG-4');
%v.FrameRate = 60;
%open(v);

for i = 1:length(T)
    plot(X, U(i,:), 'r:', 'LineWidth', 2, 'DisplayName', "Square wave");

    % plot settings
    title(sprintf('t = %.3f', T(i)));
    legend('Location', 'west');
    grid on;
    xlabel('x');
    xticks([0 pi/2 pi 3*pi/2 2*pi]);
    xticklabels({'0','\pi/2','\pi','3\pi/2','2pi'});
    xlim([0,2*pi]);
    ylabel('u(x,t)');
    ylim([min(u0)-padding,max(u0)+padding]);
    pause(0.01);
    % Write current frame to video
    %frame = getframe(f);
    %writeVideo(v, frame);
    %pause(0.01);
end
%close(v);
%disp('Animation saved as square_wave.mp4');
%% Plot at specific times for stillframes
times = [0 2 5 10];
indices = ones(1,length(times));
for i = 2:length(times)
    target = times(i);
    [~,index] = min(abs(T-target)); % retrieve index
    indices(i) = index;
end

f = figure();
f.Position = [100,100,360,240];

for i = 1:length(times)
    plot(X, U(indices(i),:), 'r:', 'LineWidth', 2, 'DisplayName', "Square wave");
    
    % plot settings
    title(sprintf('t = %.3f', times(i)));
    legend('Location', 'west');
    grid on;
    xlabel('x');
    xticks([0 pi/2 pi 3*pi/2 2*pi]);
    xticklabels({'0','\pi/2','\pi','3\pi/2','2pi'});
    xlim([0,2*pi]);
    ylabel('u(x,t)');
    ylim([min(u0)-padding,max(u0)+padding]);
    
    formatSpec = '/Figures/square_wave_%d.png';
    str = sprintf(formatSpec, times(i));
    exportgraphics(f, [pwd str], 'Resolution', 300);
    
    pause(0.05);
end