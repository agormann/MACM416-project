function [T,U] = godunov(X,u0,tf)
%GODUNOV Solves the inviscid burgers equation via the finite volume method.
% Input
%   X := grid of x values
%   u0 := initial profile
%   tf := final time
% Output
%   T := time-steps that the problem was solved at
%   U := profiles at times t in T

% Setup
f = @(u) 0.5*u.^2; % flux function
nx = length(X);
dx = abs(X(1)-X(2));
u = u0;
t = 0;
U = u;
T = t;

% Solution
% dt = dx/max(abs(u)); % fix dt for mesh refinement
while t < tf
    dt = dx/max(abs(u)); % compute dt (dynamic)
    flux = f(u); % initial flux
    F = zeros(size(u)); % flux update
    % solving riemann problem exactly
    for i = 1:nx
        uL = u(i);
        fuL = flux(i);
        uR = u(mod(i,nx)+1);
        fuR = flux(mod(i,nx)+1);
        if uL <= uR
            F(i) = min(fuL,fuR);
        elseif uL > uR
            F(i) = max(fuL,fuR);
        else
            error('OOPSIE WOOPSIE!!')
        end
    end
    u = u - dt/dx*(F - circshift(F,1)); % conservative update
    t = t + dt; U = [U; u]; T = [T; t];
end
end