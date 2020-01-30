clc
imds = imageDatastore('new asl','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9);
imageAugmenter = imageDataAugmenter(...
    'RandRotation',[-20, 20],...
    'RandXTranslation',[-3,3],...
    'RandYTranslation',[-3,3]);
imageSize = [224 224 3];
augimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);
imdsValidation.ReadFcn = @(loc)imresize(imread(loc),[224 224]);
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),[224 224]);
net = Gfinal2

net.Layers(1)
inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') %googlenet not a series network
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

% numClasses = numel(categories(imdsTrain.Labels));
numClasses = 27;
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

miniBatchSize = 10;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimds,lgraph,options);

[YPred,probs] = classify(net,imdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

Gfinal4 = net;
save Gfinal4; 