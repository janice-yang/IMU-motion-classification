clear
close all

addpath('models/')
addpath('datasets/')
addpath('functions/')

rng(0)

%% Load, visualize, & clean datasets
% Training dataset
filename = 'datasets/train/X_train.txt' ;
opts = detectImportOptions(filename) ;
opts.LeadingDelimitersRule = 'ignore' ;
opts.Delimiter = ' ' ;
Xtrain = table2array(readtable(filename, opts)) ;

filename = 'datasets/train/y_train.txt' ;
opts = detectImportOptions(filename) ;
opts.LeadingDelimitersRule = 'ignore' ;
opts.Delimiter = ' ' ;
Ytrain = table2array(readtable(filename, opts)) ;

filename = 'datasets/train/subject_train.txt' ;
opts = detectImportOptions(filename) ;
opts.LeadingDelimitersRule = 'ignore' ;
opts.Delimiter = ' ' ;
subjects_train = table2array(readtable(filename, opts)) ;

% Testing dataset
filename = 'datasets/test/X_test.txt' ;
opts = detectImportOptions(filename) ;
opts.LeadingDelimitersRule = 'ignore' ;
opts.Delimiter = ' ' ;
Xtest = table2array(readtable(filename, opts)) ;

filename = 'datasets/test/y_test.txt' ;
opts = detectImportOptions(filename) ;
opts.LeadingDelimitersRule = 'ignore' ;
opts.Delimiter = ' ' ;
Ytest = table2array(readtable(filename, opts)) ;

filename = 'datasets/test/subject_test.txt' ;
opts = detectImportOptions(filename) ;
opts.LeadingDelimitersRule = 'ignore' ;
opts.Delimiter = ' ' ;
subjects_test = table2array(readtable(filename, opts)) ;

% Dataset visualization?

% Dataset cleaning?

%% Train model
% Reference: https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-deep-learning.html
% Reference: 
model_name = 'CNN-LSTM-20250502' ;

numSamples = height(Xtrain) ;
numFeatures = width(Xtrain) ;
numHiddenUnits = 200 ;
numClasses = 6 ;

% Set up cross-validation
method = 'Kfold' ;
splits = 5 ;
indices = crossvalind('Kfold',numSamples,splits) ;
cp = classperf((Ytrain')) ;

% filterSize = ceil(numFeatures/4) ;
filterSize = 32 ;
numFilters = 32 ;

layers = [ ...
    sequenceInputLayer(numFeatures, 'Name', 'input')
    
    % CNN
    % sequenceFoldingLayer('Name','fold')
    % CNN feature extraction
    convolution1dLayer(filterSize,numFilters, 'WeightsInitializer','zeros', 'Padding', 'same') ;
    % Normalize a mini-batch of data - reduce sensitivity to initialization + speed up training 
    batchNormalizationLayer('Name','bn')
    % ELU neuron activation operation
    eluLayer('Name','elu')
    % Pooling (CNN dimension reduction)
    averagePooling1dLayer(1,'Stride',filterSize,'Name','pool1')
    % Unfolding layer
    % sequenceUnfoldingLayer('Name','unfold')

    % LSTM
    flattenLayer('Name','flatten')
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    % Softmax activation 
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',60, ...
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Training using <splits>-fold cross-validation
for i=1:splits
    % Get training & validation indices
    i_validation = (indices == i); 
    i_train = ~i_validation;
    % Training
    net = trainNetwork( ...
        num2cell(transpose(Xtrain(i_train,:)),1), ...
        categorical(transpose((Ytrain(i_train,:)))), ...
        layers, ...
        options);
    % Validation
    class = classify(net, num2cell(Xtrain(i_validation,:).', 1));
    % Calculate validation performance
    classperf(cp, double(class), i_validation);
    save(['models/', model_name, '/TrainingSplit', int2str(i), '.mat'], 'net', 'i_train', 'i_validation', 'class')
end
disp(['Final validation error: ', num2str(cp.ErrorRate)])

% Position visual + classification

% Accuracy & precision of model + benchmark comparisons?

% Store results as .mat for data & .pdf for visualizations/summaries
save(['models/' model_name, '/ClassifierPerformance.mat'], 'cp', 'net')

%% Test model
% Reference: https://www.mathworks.com/help/deeplearning/ug/sequence-to-sequence-classification-using-deep-learning.html

Ypred = classify(net, num2cell(Xtest', 1));
testing_accuracy = sum(double(Ypred) == Ytest)/numel(Ytest) ;

% Accuracy & precision of model + benchmark comparisons?

% Store results as .mat for data & .pdf for visualizations/summaries
save(['models/' model_name, '/ClassifierTesting.mat'], 'Xtest', 'Ytest', 'Ypred', 'testing_accuracy')

%% Simulate new data

