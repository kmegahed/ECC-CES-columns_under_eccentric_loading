uqlab; clear; fclose all; clc;
% Load the MAT-file if it exists
matFileName = 'C:\Users\osama\OneDrive\Desktop\abaqus2\X_Y_data1_m.mat';

% Define your model
ModelOpts.Name = 'my_model_,';
ModelOpts.mFile = 'model_m';%[30;30],[2;2.22],[40;30.1],[240;300]
ModelOpts.isVectorized = true;
myModel = uq_createModel(ModelOpts);

% Define input parameters
II.Marginals(1).Type = 'Uniform';II.Marginals(1).Parameters = [300 600];%B
II.Marginals(2).Type = 'Uniform';II.Marginals(2).Parameters = [300 1200];%H
II.Marginals(3).Type = 'Uniform';II.Marginals(3).Parameters = [4 40];%Lam
II.Marginals(4).Type = 'Uniform';II.Marginals(4).Parameters = [235 460];%fy_rebar
II.Marginals(5).Type = 'Uniform';II.Marginals(5).Parameters = [20 50];%fc
II.Marginals(6).Type = 'Uniform';II.Marginals(6).Parameters = [235 460];%fs_steel
II.Marginals(7).Type = 'Uniform';II.Marginals(7).Parameters = [0.25 0.8];%d_steel (Asfy/Arebar fy +0.85*fc*Ac)
II.Marginals(8).Type = 'Uniform';II.Marginals(8).Parameters = [0.01 0.04];%ro_rebar
II.Marginals(9).Type = 'Uniform';II.Marginals(9).Parameters = [2.0 4.0];%tf_(0.5*tw)
II.Marginals(10).Type= 'Uniform';II.Marginals(10).Parameters= [40.0 100.0];%cH1
II.Marginals(11).Type= 'Uniform';II.Marginals(11).Parameters= [40.0 100.0];%cB
II.Marginals(12).Type= 'Uniform';II.Marginals(12).Parameters= [0.0 2.389];%ecc

%B=300.0;H=600.0;lam=5.0;fy=230.0;fe=25.0;    fs=250.0;d_steel=0.3;ro_rebar=0.01;tf_tw=1.50;cH=60;cB=50;

I = uq_createInput(II);

% Specify the objective
ObjectiveOpts.Type = 'UQLink';
ObjectiveOpts.Model = myModel;


% Perform initial experiments
if exist(matFileName, 'file') == 2
    load(matFileName, 'X_all', 'Y_all');
    X=X_all;Y=Y_all;
else
    % Perform initial experiments
    X = uq_getSample(I, 3, 'LHS');
    Y = uq_evalModel(myModel, X);
    
    % Save X and Y to a MAT-file
    save(matFileName, 'X', 'Y');
end


% Update the model with the observed data
MetaOpts.Type = 'Metamodel';
MetaOpts.MetaType = 'KRiging';
MetaOpts.Input = I;
MetaOpts.ExpDesign.X = X;
MetaOpts.ExpDesign.Y = Y;
MetaOpts.Display = 'quiet';
MetaOpts.Optim.Method = 'none'; % Skip hyperparameters optimization for simplicity
metaModel = uq_createModel(MetaOpts);

% Define the number of iterations
num_iterations = 500;

% Initialize arrays to store results for plotting
X_all = X;
Y_all = Y;
errors_all = zeros(num_iterations+1, 1);
check1=[];
% Perform sequential optimization
Y_all11=[];
for iter = 1:num_iterations
    % Select the next experimental point
    NextPoint = mySelectSample(I, metaModel);

    % Perform the experiment and obtain the response
    [Y_new,check] = uq_evalModel(myModel, NextPoint);

    % Update the experimental design and response arrays
    X_all = [X_all; NextPoint];
    Y_all = [Y_all; Y_new];
    Y_all11 = [Y_all11;Y_new]
    check1 = [check1; check]
    save(matFileName, 'X_all', 'Y_all');

    % Update the surrogate model with the new data
    MetaOpts.ExpDesign.X = X_all;
    MetaOpts.ExpDesign.Y = Y_all;
    metaModel = uq_createModel(MetaOpts);

    % Store the errors for plotting
    [~, errors] = uq_evalModel(metaModel, X_all);
    errors_all(iter) = errors(end);

    % Display the current iteration
    fprintf('Iteration %d completed.\n', iter);
    % Plot the results
%{
figure;
subplot(2, 1, 1);
plot(X_all(:, 1), Y_all, 'bo', 'MarkerSize', 8); hold on;
x_true = linspace(0, 1, 100);
plot(x_true, x_true.^2, 'r-', 'LineWidth', 2);
xlabel('x');
ylabel('y');
title('True Function and Chosen Points');
legend('Chosen Points', 'True Function');
grid on; grid minor;

subplot(2, 1, 2);
plot(1:iter, errors_all(1:iter), 'bs-', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Error');
title('Error Evolution');
grid on; grid minor;
%}
end
figure;
subplot(2, 1, 1);
plot(X_all(:, 1), Y_all, 'bo', 'MarkerSize', 8); hold on;
x_true = linspace(0, 1, 100);
plot(x_true, x_true.^2, 'r-', 'LineWidth', 2);
xlabel('x');
ylabel('y');
title('True Function and Chosen Points');
legend('Chosen Points', 'True Function');
grid on; grid minor;

subplot(2, 1, 2);
plot(1:num_iterations, errors_all(1:num_iterations), 'bs-', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Error');
title('Error Evolution');
grid on; grid minor;

%{
uqlab;clear;fclose all;clc;
% Define your model
ModelOpts.Name = 'my_model';
ModelOpts.mString = 'X(:,1).^2';
ModelOpts.isVectorized = true;
myModel = uq_createModel(ModelOpts);

% Define input parameters
InputOpts.Marginals(1).Type = 'Uniform';
InputOpts.Marginals(1).Parameters = [0 1];
I=uq_createInput(InputOpts);
% Create a UQLab model

% Specify the objective
ObjectiveOpts.Type = 'UQLink';
ObjectiveOpts.Model = myModel;

% Define the experimental design options
ExpDesignOpts.Type = 'Latin Hypercube';
ExpDesignOpts.NSamples = 10;

% Perform initial experiments
X = uq_getSample(I,10,'LHS');

% Evaluate the model at the initial experimental points
Y = uq_evalModel(myModel, X);

% Update the model with the observed data
MetaOpts.Type = 'Metamodel';
MetaOpts.MetaType = 'KRiging';
MetaOpts.Input = I;
MetaOpts.ExpDesign.X = X;
MetaOpts.ExpDesign.Y = Y;
MetaOpts.Display = 'quiet';
MetaOpts.Optim.Method = 'none'; % Skip hyperparameters optimization for simplicity
metaModel = uq_createModel(MetaOpts);
% Define the number of iterations
num_iterations = 10;

% Perform sequential optimization
for iter = 1:num_iterations
    % Select the next experimental point
    NextPoint = mySelectSample(I,metaModel);

    % Perform the experiment and obtain the response
    % (Replace this with your actual experiment/simulation code)
    % For demonstration purposes, let's assume a simple quadratic response
    Y_new = uq_evalModel(myModel, NextPoint);%my_model_function(NextPoint);

    % Update the experimental design and response arrays
    X = [X; NextPoint];
    Y = [Y; Y_new];

    % Update the surrogate model with the new data
    MetaOpts.ExpDesign.X = X;
    MetaOpts.ExpDesign.Y = Y;
    metaModel = uq_createModel(MetaOpts);

    % Display the current iteration
    fprintf('Iteration %d completed.\n', iter);
end
%}
function NextPoint = mySelectSample(iii, metaModel)
    % Generate a set of candidate points
    num_candidates = 100; % Adjust as needed
    candidates = uq_getSample(iii, num_candidates);

    % Evaluate the surrogate model at the candidate points
    [gmean, gsigma2] = uq_evalModel(metaModel, candidates);
    gsigma = gsigma2.^0.5;

    % Set the current best observed value (y_best)
    y_best = min(gmean); % Adjust based on your optimization goal (minimization)

    % Calculate the improvement over the current best value
    improvement = y_best - gmean;
    
    % Calculate the standard Expected Improvement (EI)
    Z = improvement ./ gsigma;
    EI = improvement .* normcdf(Z) + gsigma .* normpdf(Z);

    % Select the candidate point with the highest Expected Improvement
    [~, idx] = max(EI);
    NextPoint = candidates(idx,:);
end
function NextPoint = mySelectSample1(iii,metaModel)
    % Generate a set of candidate points
    num_candidates = 100; % Adjust as needed
    candidates = uq_getSample(iii, num_candidates);

    % Evaluate the surrogate model at the candidate points
    %[pred,errorss] = uq_evalModel(metaModel, candidates);

    % Select the candidate point with the highest predicted value
    %[~, idx] = max(errorss.^0.5./(pred+0.1));
    [gmean,gsigma2] = uq_evalModel(metaModel, candidates);
    gsigma=gsigma2.^0.5;    eps = 2*gsigma;gmean=gmean+0.1;
[~, idx] = max(gmean .*  ( 2*normcdf(-gmean./gsigma, 0, 1) - normcdf(-(eps+gmean)./gsigma, 0, 1) - normcdf((eps-gmean)./gsigma, 0, 1))...
    -gsigma .* ( 2*normpdf(-gmean./gsigma, 0, 1) - normpdf(-(eps+gmean)./gsigma, 0, 1) - normpdf((eps-gmean)./gsigma, 0, 1))...
    +eps .* ( normcdf((eps-gmean)./gsigma, 0, 1) - normcdf((-eps-gmean)./gsigma, 0, 1)));
    NextPoint = candidates(idx,:);
end

function [lf, idx, xadded] = uq_LF_CMM(gmean,xcandidate,XED, K)
% UQ_LF_CMM computes an adpatation of the constrained min-max criterion
% developed in the following reference:
%
% Get the closest points to the LSF
Nout = size(gmean,2);
% Select the 1% closest points (should this be an option?)
Nselect = 0.01 * size(xcandidate,1) ;
xadded = [] ;



for j = 1: Nout
    % Sort the points according to their distance to the limit-state surface
    [sortedXG, idXG] = sortrows([xcandidate,gmean(:,j)],size(xcandidate,2)+1, 'ascend', ...
        'ComparisonMethod','abs') ;
    xselect = sortedXG(1:Nselect,1:size(xcandidate,2)) ;
    
    
    for kk = 1:1
        % Get the nearest neighbour of the training points for each of the points
        % in the selected enrichment candidates
        [idx, dist_to_XED] = knnsearch([XED;xadded],xselect) ;
        idx = idXG(idx) ;
        
        % Learning function
        [lf, indlf] = max(dist_to_XED,[],1);
        indlf
        % index of the added point
        idx = idXG(indlf) ;
        xnext = xcandidate(idx,:);
        xadded = [xadded; xnext] ;
        
    end
    
end
end



%force=model([30;30],[2;2.22],[40;30.1],[240;300])


%{
% Define the file path
file_path = 'C:\Users\osama\OneDrive\Desktop\Job-1.dat'; % Replace 'path_to_your_file.txt' with the actual file path

% Open the file for reading
fid = fopen(file_path, 'r');
if fid == -1
    error('Failed to open file.');
end

% Read the contents of the file
textData = fread(fid, '*char')';
fclose(fid);

% Define the regular expression pattern to match the desired data pattern
pattern = '^\s+\d+\s+(\d+\.\d+)\s+(\d+\.\d+E[+\-]\d+)\s*$';

% Find matches for the pattern in the text
matches = regexp(textData, pattern, 'tokens', 'lineanchors');

% Extract the matched data
matchedData = cellfun(@(x) [str2double(x{1}), str2double(x{2})], matches, 'UniformOutput', false);

% Convert to a numeric matrix
dataMatrix = cell2mat(matchedData');

% Display the extracted data
disp('Extracted data:');
disp(dataMatrix);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define regular expressions to extract U3 and RF3 values
u3Pattern = 'U3\s+(\S+)';
rf3Pattern = 'RF3\s+(\S+)';

% Find matches for U3 and RF3 values
u3Matches = regexp(textData, u3Pattern, 'tokens');
rf3Matches = regexp(textData, rf3Pattern, 'tokens');

% Extract numerical values from the matches
u3Values = cellfun(@(x) str2double(x{1}), u3Matches);
rf3Values = cellfun(@(x) str2double(x{1}), rf3Matches);

% Display the extracted data
disp('U3 values:');
disp(u3Values);
disp('RF3 values:');
disp(rf3Values);



% Define the file path
file_path = 'C:\Users\osama\OneDrive\Desktop\abaqus.py';

% Open the file for reading
fid = fopen(file_path, 'r');
if fid == -1
    error('Failed to open file.');
end

% Read the contents of the file
contents = fscanf(fid, '%c');
fclose(fid);

% Find and replace the depth value
contents = strrep(contents, 'p.BaseSolidExtrude(sketch=s, depth=40.0)', 'p.BaseSolidExtrude(sketch=s, depth=30.0)');

% Open the file for writing
fid = fopen(file_path, 'w');
if fid == -1
    error('Failed to open file for writing.');
end

% Write the modified contents back to the file
fprintf(fid, '%s', contents);
fclose(fid);

disp('Replacement completed successfully.');
%}
