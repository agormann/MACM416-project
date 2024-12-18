function [T,U] = godunov_dirichlet(X,u0,tf)
%GODUNOV_DIRICHLET Solves the inviscid burgers equation via the finite volume method. 
% Dirichlet boundary conditions are assumed.
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
while t < tf
    dt = dx/max(abs(u)); % compute dt (dynamic)
    flux = f(u); % initial flux
    F = zeros(size(u)); % flux update
    % solving riemann problem exactly
    for i = 1:nx-1
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
    for i = 2:nx-1
        u(i) = u(i) - dt/dx*(F(i)-F(i-1)); % conservative update
    end
    t = t + dt; U = [U; u]; T = [T; t];
end
end