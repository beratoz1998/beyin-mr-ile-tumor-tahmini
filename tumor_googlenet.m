clear all
close all
clc

% DATA LOADING

imds = imageDatastore('dataset', 'IncludeSubfolders',true, 'LabelSource','foldernames');
 imds.ReadFcn=@readFCN;
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomized');

% CREATE ALEXNET & LAYERS
net = googlenet;
lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

% TRANSFER LEARNING

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);


% AUGMENTATION OF TRAINING DATA

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% VALIDATION AND TEST DATA 
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


%TRAIN NETWORK WITH OPTIONS

options = trainingOptions('sgdm', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',25, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);
% testset=preview(augimdsTest);
% testimage=testset{1,1}{1};
% test_gt=testset{1,2};
% label = classify(netTransfer,testimage);
% figure
% imshow(testimage)
% title(string(label))
% PERFORMANCE CHECK FOR VALIDATION & TEST DATA

[YPredval,scoresval] = classify(netTransfer,augimdsValidation);

confMatval = confusionmat(imdsValidation.Labels, YPredval);
confMatval = confMatval./sum(confMatval,2)
mean(diag(confMatval))












