% PINN FOR INVISCID BURGERS (u0 = {0 if x < 0, 1 if x > 0})
% Adapted from: https://www.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html

%% TRAINING DATA GENERATION
    
% Parameters                    
numBoundaryConditionPoints = [25, 25];
numInitialConditionPoints = 50;
numInternalCollocationPoints = 10000;

% Time points to enforce boundary conditions
x0BC1 = -pi * ones(1, numBoundaryConditionPoints(1));        % x vals on left
x0BC2 = pi * ones(1, numBoundaryConditionPoints(2));    % x vals on right

t0BC1 = linspace(0, 4, numBoundaryConditionPoints(1)); % times
t0BC2 = linspace(0, 4, numBoundaryConditionPoints(2)); 

u0BC1 = zeros(1, numBoundaryConditionPoints(1));         % left = 0
u0BC2 = ones(1, numBoundaryConditionPoints(2));        % right = 1

% Spatial points to enforce initial conditions
x0IC = linspace(-pi, pi, numInitialConditionPoints);  % Spatial points in [-pi, pi]
t0IC = zeros(1, numInitialConditionPoints);           % Time = 0
u0IC = double(x0IC > 0);                             % Initial condition: 0 for x < 0, 1 otherwise

% Group together boundary and initial condition data
X0 = [x0IC, x0BC1, x0BC2];
T0 = [t0IC, t0BC1, t0BC2];
U0 = [u0IC, u0BC1, u0BC2];

% Uniformly sample 10 000 points (t,x) ∈ (0,10) × (0,2pi)
% to enforce the output of the network to fulfill the PDE
pointSet = sobolset(2);
points = net(pointSet,numInternalCollocationPoints);

dataX = 2*pi*points(:,1)-pi;
dataT = 4*points(:,2);

% array datastore containing training data
ds = arrayDatastore([dataX dataT]);

%% DEFINE DEEP LEARNING MODEL

% NN parameters
numLayers = 9;
numNeurons = 20;

% init paras for first fully connect operation (?)
parameters = struct;

sz = [numNeurons 2];
parameters.fc1_Weights = initializeHe(sz,2,'double');
parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');

% init paras for remaining intermediate fully connect ops (?)
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz,numIn,'double');
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end

% init paras for final fully connect operation (?)
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,'double');
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');

%% SPECIFY OPTIMIZATION OPTIONSF

options = optimoptions('fmincon', ...
    'HessianApproximation','lbfgs', ...
    'MaxIterations',5000, ...
    'MaxFunctionEvaluations',5000, ...
    'OptimalityTolerance',1e-5, ...
    'SpecifyObjectiveGradient',true);

%% TRAIN NN

[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);

%convert training data to dlarray objects
dlX = dlarray(dataX','CB');
dlT = dlarray(dataT','CB');
dlX0 = dlarray(X0,'CB');
dlT0 = dlarray(T0,'CB');
dlU0 = dlarray(U0,'CB');

% objective func definition
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,dlX0,dlT0,dlU0,parameterNames,parameterSizes);

% update learnable parameters
parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);

% for prediction, convert paras to struct
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

%% EVALUATE MODEL ACCURACY

% parameters (test times, spacial points)
tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);

%test model
figure
for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);

    % Make predictions.
    dlXTest = dlarray(XTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = model(parameters,dlXTest,dlTTest);

    % Calcualte true values.
    UTest = solveBurgers(XTest,t); % modified to remove 3rd parameter

    % Calculate error.
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest);

    % Plot predictions.
    subplot(2,2,i)
    plot(XTest,extractdata(dlUPred),'-','LineWidth',2);
    ylim([-1.1, 1.1])

    % Plot true values.
    hold on
    plot(XTest, UTest, '--','LineWidth',2)
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

subplot(2,2,2)
legend('Predicted','True')

%% MOVIE

% movie parameters
numFrames = 100;                
timeSteps = linspace(0, 4, numFrames); 
numPredictions = 1001;
XMovie = linspace(-pi, pi, numPredictions);

% video stuff
videoFileName = 'moviviv.mp4';
videoWriter = VideoWriter(videoFileName, 'MPEG-4');
videoWriter.FrameRate = 10;
open(videoWriter);

figure;
for t = timeSteps

    % plot approximation
    TMovie = t * ones(1, numPredictions);
    dlXMovie = dlarray(XMovie, 'CB');
    dlTMovie = dlarray(TMovie, 'CB');
    dlUPred = model(parameters, dlXMovie, dlTMovie);
    plot(XMovie, extractdata(dlUPred), 'b-', 'LineWidth', 2);
    hold on;

    % plot true solutionm
    Utrue = solveBurgers(XMovie, t);
    plot(XMovie, Utrue, 'r--', 'LineWidth', 2);
    hold off;

    % Calculate error.
    err = norm(extractdata(dlUPred) - Utrue);
    relErr = norm(extractdata(dlUPred) - Utrue) / norm(Utrue);

    %plot info stuff
    title(['t = ', num2str(t), ', Error_{abs} = ', num2str(err), ', Error_{rel} = ', num2str(relErr)]);
    xlabel('x');
    ylabel('u(x, t)');
    legend('Predicted', 'True');
    ylim([-0.1, 1.1]);
    grid on;

    % write frames
    frame = getframe(gcf);
    writeVideo(videoWriter, frame);
end

close(videoWriter);

%% SOLVE INV BURGERS FUNC

function U = solveBurgers(X, t)

    uL = 0;  
    uR = 1;  
    U = zeros(size(X));

    for i = 1:numel(X)
        x = X(i);

        if x <= uL * t 
            U(i) = uL;  
        elseif x > uR * t
            U(i) = uR;
        else
            U(i) = uL + (uR - uL) * (x / t);
        end
    end
end

%% OBJECTIVE FUNC

function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlT,dlX0,dlT0,dlU0,parameterNames,parameterSizes)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlT,dlX0,dlT0,dlU0);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end

%% MODEL GRADIENTS FUNC

function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlX0,dlT0,dlU0)

% Make predictions with the initial conditions.
U = model(parameters,dlX,dlT);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions.
dlU0Pred = model(parameters,dlX0,dlT0);
lossU = mse(dlU0Pred, dlU0);

% Combine losses.
loss = lossF + lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end

%% MODEL FUNC

function dlU = model(parameters,dlX,dlT)

dlXT = [dlX;dlT];
numLayers = numel(fieldnames(parameters))/2;

% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
dlU = fullyconnect(dlXT,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    dlU = tanh(dlU);

    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    dlU = fullyconnect(dlU, weights, bias);
end

end