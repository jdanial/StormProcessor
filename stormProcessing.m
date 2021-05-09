%% Copyright 2020 John S H Danial
%% Department of Chemistry, Univerity of Cambridge
%%
%% GPU Coder library has to be installed
%% Deep Learning library has to be installed

%% --------- DO NOT EDIT THIS SECTION --------- %%

clear all;
clc;
folder = fileparts(which('stormProcessing.m'));
addpath(genpath(folder));

%% -------------------------------------------- %%

%%  --------- modifiable inputs --------- %%

%% extract & process (0) or processOnly (1)
mode = 0;

%% file type (tif or dat)
fileType = 'tif';

%% live analysis
liveAnalysis = false;

%% number of files (x1000)
startFrame = 1000;
endFrame = 100000;

%% setting ROI radius
roiRadius = 5;

%% camera parameters (set cameraEMGain to 1 if using sCMOS)
cameraPixelSize = 117;
cameraOffset = 400;
cameraEMGain = 300;
cameraConversionFactor = 3.6;
cameraQE = 0.9;

%% image pixel size
imagePixelSize = 5;

%% localization filtering parameters (in nanometers)
maxSigma = 150;
maxLocPrec = 10;

%% perform drift correction (none, beads or crossCorr)
driftCorrection = 'crossCorr';

%% group localizations (maxGap [in frames], maxDist [in nanometers])
groupLoc = true;
maxGap = 1;
maxDist = 35;

%% --------- END OF MODIFIABLE INPUTS --------- %%

%% --------- DO NOT EDIT FROM HERE --------- %%

%% get folder containing images
fPath = uigetdir();

%% deleting super resolved image if exists
if mode == 0
    delete(fullfile(fPath,'SR.tif'));
    delete(fullfile(fPath,'SR_proc.tif'));
end

%% getting list of images
if mode == 0
    fileList = dir(fullfile(fPath, ['*.' fileType]));
end

%% calculating parameters
if mode == 0
    roiWidth = roiRadius * 2 + 1;
end

%% calculating magnification factor
magFactor = cameraPixelSize / imagePixelSize;

%% setting up particle image grid for bkgd calculation
if mode == 0
    rowsLin = repmat(roiWidth : -1 : 1,1,roiWidth);
    colsTempLin = repmat(1:roiWidth,roiWidth,1);
    colsLin = colsTempLin(:);
end

%% initializing row and col drift
if mode == 0
    allData = zeros(10000000,5);
end

%% initializing SR image
if mode == 0
    if strcmp(fileType,'dat')
        fName = dir(fullfile(fPath,'*.ini'));
        paramFile = fopen(fullfile(fPath,fName),'r');
        line = fgetl(paramFile);
        while (~(line == -1))
            paramName= line(1 : (strfind(line,'=') - 1));
            paramValue = line((strfind(line,'=') + 1) : end);
            if (contains(paramName,'AOIWidth'))
                width = str2double(paramValue);
            elseif (contains(paramName,'AOIHeight'))
                height = str2double(paramValue);
            end
            line = fgetl(paramFile);
        end
        imageSR = zeros(height .* magFactor,width .* magFactor);
        imageSR_proc = imageSR;
    else
        fName = fileList(1).name;
        imageRaw = imread(fullfile(fPath,fName));
        imageSR = zeros(round(size(imageRaw) .* magFactor));
        imageSR_proc = imageSR;
    end
end

%% dictating numeric file type
if mode == 0
    if strcmp(fileType,'tif')
        fileTypeNum = 0;
    else
        fileTypeNum = 1;
    end
end

%% loading neural network
if mode == 0
    net = load('neuralNetwork.mat');
    neuralNetwork = net.neuralNet;
end

%% creating waitbar
hWaitbar = waitbar(0 , ...
    '', ...
    'Name', ...
    'Processing Stack', ...
    'CreateCancelBtn', ...
    'delete(gcbf)');

%% initializing other parameters
if mode == 0
    firstBatch = true;
    fInd = 1;
end

%% looping through batches (each batch 1000 frames)
partAcc = 0;
if mode == 0
    for batchId = floor(startFrame / 1000) + 1 : ceil(endFrame / 1000)
        
        %% check batch acquired
        if liveAnalysis
            acquiring = true;
            while acquiring
                waitbar(fileId / endFrame, ...
                    hWaitbar, ...
                    'Acquiring ...');
                pause(1);
                fileList = dir(fullfile(fPath, ['*.' fileType]));
                if length(fileList) > batchId * 1000
                    acquiring = false;
                end
                if ~ishandle(hWaitbar)
                    break;
                end
            end
        end
        if ~ishandle(hWaitbar)
            break;
        end
        
        %% initializing vectors
        imageRawVec = zeros(roiWidth,roiWidth,1,10000000);
        subImage = zeros(roiWidth,roiWidth,10000000);
        fitDataTemp = zeros(roiWidth ^ 2,10000000);
        fitInitParamsTemp = zeros(5,10000000);
        rowPosTemp = zeros(10000000,1);
        colPosTemp = zeros(10000000,1);
        frameTemp = zeros(1,10000000);
        globalParticleId = 1;
        
        for fileId = ((batchId - 1) * 1000) + 1 : min([batchId * 1000 batchId * endFrame])
            
            %% displaying message
            waitbar(fileId / endFrame, ...
                hWaitbar, ...
                ['File ' num2str(fileId) ' out of ' num2str(endFrame) ' - extracting particles']);
            
            %% reading imageRaw
            fName = fileList(fileId).name;
            if fileTypeNum == 0
                imageRaw = ((double(imread(fullfile(fPath,fName))) - cameraOffset) ./ (cameraEMGain / cameraConversionFactor)) ./ cameraQE;
            else
                imageFile = fopen(fullfile(fPath,fName),'r');
                imageTemp = fread(imageFile,width * height,'uint32=>uint32');
                fclose(imageFile);
                imageRaw = ((double(reshape(imageTemp,width,height)) - cameraOffset) ./ (cameraEMGain / cameraConversionFactor)) ./ cameraQE;
            end
            
            %% thresholding imageRaw
            thresholdImageRaw = imregionalmax(imageRaw,8);
            
            %% extracting connected objects from threshold imageRaw
            [centroidTemp_y,centroidTemp_x] = ind2sub(size(thresholdImageRaw),find(thresholdImageRaw == true));
            centroids = [centroidTemp_x centroidTemp_y];
            
            %% creating an array of all particles in an imageRaw
            indices = find(centroids(:,1) > 2 + roiRadius & ...
                centroids(:,1) < size(imageRaw,2) - roiRadius - 1 & ...
                centroids(:,2) > 2 + roiRadius & ...
                centroids(:,2) < size(imageRaw,1) - roiRadius - 1);
            for particleId = 1 : length(indices)
                subImage(:,:,particleId) = imageRaw(round(centroids(indices(particleId),2)) + roiRadius : -1 :...
                    round(centroids(indices(particleId),2)) - roiRadius,...
                    round(centroids(indices(particleId),1)) - roiRadius:...
                    round(centroids(indices(particleId),1)) + roiRadius);
            end
            maxSubImage = max(subImage(:,:,1 : length(indices)),[],[1,2]);
            minSubImage = min(subImage(:,:,1 : length(indices)),[],[1,2]);
            rowPosTemp(globalParticleId : globalParticleId + length(indices) - 1,1) = ...
                round(centroids(indices,2)) + roiRadius;
            colPosTemp(globalParticleId : globalParticleId + length(indices) - 1,1) = ...
                round(centroids(indices,1)) - roiRadius;
            fitDataTemp(:,globalParticleId : globalParticleId + length(indices) - 1) = ...
                reshape(subImage(:,:,1 : length(indices)),[roiWidth * roiWidth,length(indices)]);
            fitInitParamsTemp(:,globalParticleId : globalParticleId + length(indices) - 1) = ...
                [reshape(maxSubImage(1,1,1:length(indices)),1,[]) - reshape(minSubImage(1,1,1:length(indices)),1,[]) ; ...
                4 * ones(1, length(indices)); ...
                4 * ones(1, length(indices)) ; ...
                1 * ones(1, length(indices)) ; ...
                reshape(minSubImage(1,1,1:length(indices)),1,[])];
            imageRawVec(:,:,1,globalParticleId : globalParticleId + length(indices) - 1) = ...
                reshape(double((subImage(:,:,1:length(indices)) - minSubImage(1,1,1:length(indices)))) ./ ...
                double((maxSubImage(1,1,1:length(indices)) - minSubImage(1,1,1:length(indices)))) ,...
                roiWidth,roiWidth,1,length(indices));
            frameTemp(1,globalParticleId : globalParticleId + length(indices) - 1) = fileId *...
                ones(1, length(indices));
            globalParticleId = globalParticleId + length(indices);
            if ~ishandle(hWaitbar)
                break;
            end
        end
        
        %% exiting software
        if ~ishandle(hWaitbar)
            break;
        end
        
        %% displaying message
        waitbar(fileId / endFrame, ...
            hWaitbar, ...
            ['File ' num2str(fileId) ' out of ' num2str(endFrame) ' - classifying particles']);
        
        %% loading trained neural network
        classVec = classify(neuralNetwork,...
            imageRawVec(:,:,:,1:globalParticleId - 1), ...
            'ExecutionEnvironment','gpu', ...
            'MiniBatchSize',100000, ...
            'Acceleration','auto');
        
        %% displaying message
        waitbar(fileId / endFrame, ...
            hWaitbar, ...
            ['File ' num2str(fileId) ' out of ' num2str(endFrame) ' - fitting particles']);
        
        %% fitting positively classified particles
        indices = classVec(:,1) == '1';
        partAcc = partAcc + sum(indices);
        fitData = single(fitDataTemp(:,indices));
        fitInitParams = single(fitInitParamsTemp(:,indices));
        rowPos = rowPosTemp(indices);
        colPos = colPosTemp(indices);
        frame = frameTemp(indices);
        [parameters, states, chi_squares, n_iterations, time] = ...
            cpufit(fitData, ...
            [], ...
            ModelID.GAUSS_2D, ...
            fitInitParams, ...
            [], ...
            [], ...
            [], ...
            []);
        
        %% calculating localization precision
        for particleId = 1 : size(fitData,2)
            background = std(fitData(:,particleId) - ...
                (parameters(5,particleId) + (parameters(1,particleId) .* ...
                exp(-((colsLin(:) - parameters(3,particleId)) .^ 2 + ...
                (rowsLin(:) - parameters(2,particleId)) .^ 2) ./ ...
                (2 .* (parameters(4,particleId)) .^ 2)))));
            locPrec_1 =  ((parameters(4,particleId) .* cameraPixelSize) .^ 2) ./ sum(fitData(:,particleId));
            locPrec_2 = (cameraPixelSize ^ 2) ./ (12 .* sum(fitData(:,particleId)));
            locPrec_3 = (4 .* sqrt(pi) .* ((parameters(4,particleId) .* cameraPixelSize) .^ 3) .* (background .^ 2)) ./ ...
                ((cameraPixelSize) .* (sum(fitData(:,particleId)) .^ 2));
            locPrec(particleId) = sqrt(locPrec_1 + locPrec_2 + locPrec_3);
        end
        
        %% adding fitted particles
        indices = states == 0 & parameters(4,:) .* cameraPixelSize > 100 & parameters(4,:) .* cameraPixelSize < maxSigma & locPrec(1 : length(states)) < maxLocPrec;
        intStack = parameters(1,indices)';
        sigStack = parameters(4,indices) .* magFactor;
        rowPosStackTemp = (rowPos(indices) - parameters(2,indices)') .* magFactor;
        colPosStackTemp = (colPos(indices) + parameters(3,indices)') .* magFactor;
        frameStackTemp = frame(indices)';
        locPrecStackTemp = locPrec(indices)' ./ imagePixelSize;
        
        %% saving fitted particles
        allData(fInd : fInd + sum(indices) - 1,:) = ...
            [frameStackTemp ...
            (colPosStackTemp ./ magFactor) .* cameraPixelSize ...
            (rowPosStackTemp ./ magFactor) .* cameraPixelSize ...
            locPrecStackTemp .* imagePixelSize ...
            intStack];
        fInd = fInd + sum(indices);
        
        %% rendering ungrouped and non-drift-corrected SR image
        for particleId = 1 : length(rowPosStackTemp)
            rows = round(rowPosStackTemp(particleId)) - 5 : round(rowPosStackTemp(particleId)) + 5;
            cols = round(colPosStackTemp(particleId)) - 5 : round(colPosStackTemp(particleId)) + 5;
            try
                for rowId = rows
                    for colId = cols
                        imageSR(rowId,colId) = imageSR(rowId,colId) + ...
                            intStack(particleId) .* exp(-((colId - colPosStackTemp(particleId)) .^ 2 + (rowId - rowPosStackTemp(particleId)) .^ 2) ./ ...
                            (2 * (locPrecStackTemp(particleId) ^ 2)));
                    end
                end
            catch
            end
        end
        
        %% displaying image
        if firstBatch
            figure;
            firstBatch = false;
        end
        imagesc(imadjust(imageSR));
        axis equal;
        colormap hot
        drawnow;
    end
    
    %% creating waitbar
    hWaitbar = waitbar(0 , ...
        '', ...
        'Name', ...
        'Processing Stack', ...
        'CreateCancelBtn', ...
        'delete(gcbf)');
    
    %% displaying message
    waitbar(fileId / endFrame, ...
        hWaitbar, ...
        'Saving unprocessed SR image');
    
    %% saving ungrouped and non-drift-corrected image
    imwrite(uint16(imageSR),fullfile(fPath,'SR.tif'));
    
    %% displaying message
    waitbar(fileId / endFrame, ...
        hWaitbar, ...
        'Saving unprocessed localization data');
    
    %% saving ungrouped and non-drift-corrected localization data (frame,xPos,yPos,locPrec)
    writematrix(allData(1 : fInd - 1,:),fullfile(fPath,'SR.txt'))
    
    %% localization filtering parameters (in nanometers)
    writematrix([cameraPixelSize ...
        cameraOffset ...
        cameraEMGain ...
        cameraQE ...
        imagePixelSize ...
        maxSigma ...
        maxLocPrec ...
        magFactor ...
        size(imageRaw,1) ...
        size(imageRaw,2)], ...
        fullfile(fPath,'Param.txt'))
end

%% reading localization parameters file
allParam = readmatrix(fullfile(fPath,'Param.txt'));
cameraPixelSize = allParam(1);
cameraOffset = allParam(2);
cameraEMGain = allParam(3);
cameraQE = allParam(4);
imagePixelSize = allParam(5);
maxSigma = allParam(6);
maxLocPrec = allParam(7);
magFactor = allParam(8);
imageRaw = zeros(allParam(9),allParam(10));

%% displaying message
waitbar(1 / 5, ...
    hWaitbar, ...
    'Loading localization data');

%% reading localization data file
allData = readmatrix(fullfile(fPath,'SR.txt'));

%% displaying message
waitbar(2 / 5, ...
    hWaitbar, ...
    'Correcting drift');

%% drift correcting localizations
if strcmp(driftCorrection,'crossCorr')
    [coordscorr, ~, ~] = RCC([...
        allData(:,2) ./ cameraPixelSize ...
        allData(:,3) ./ cameraPixelSize ...
        allData(:,1)], ...
        (allData(end,1) - allData(1,1)) / 20, ...
        max(size(imageRaw)), ...
        cameraPixelSize, ...
        imagePixelSize, ...
        0.2);
    allData(:,2) = coordscorr(:,1)' .* cameraPixelSize;
    allData(:,3) = coordscorr(:,2)' .* cameraPixelSize;
end

%% displaying message
waitbar(3 / 5, ...
    hWaitbar, ...
    'Grouping localizations');

%% grouping localizations
if groupLoc
    particleId = 1;
    arraySize = size(allData,1);
    while particleId < arraySize - 1000
        particleId = particleId + 1;
        previousFrame = allData(particleId,1);
        currentGap = 0;
        groupParticles = true;
        nextFramesIndices = find(allData(particleId : particleId + 1000,1) >= allData(particleId,1) + 1 & allData(particleId : particleId + 1000,1) <= allData(particleId,1) + 1 + maxGap);
        replicas = sum(sqrt(((allData(particleId,2) - allData(nextFramesIndices,2)) .^ 2) + ...
            ((allData(particleId,3) - allData(nextFramesIndices,3)) .^ 2)) < maxDist);
        if replicas > 0
            nextFramesIndices = find(allData(particleId:end,1) >= allData(particleId,1) + 1 & allData(particleId:end,1) <= allData(particleId,1) + 100);
            indices = find(sqrt(((allData(particleId,2) - allData(nextFramesIndices,2)) .^ 2) + ...
                ((allData(particleId,3) - allData(nextFramesIndices,3)) .^ 2)) < maxDist);
            if ~isempty(indices)
                for index = 1 : length(indices)
                    currentFrame = allData(indices(index),1) - currentGap;
                    if currentFrame ~= previousFrame + index
                        currentGap = currentGap + 1;
                    end
                    if currentGap > maxGap
                        groupParticles = false;
                        break;
                    end
                end
            end
            if groupParticles
                allData(particleId,2:5) = mean([allData(particleId,2:5);allData(indices,2:5)]);
                allData(indices,:) = [];
            end
        end
        arraySize = size(allData,1);
    end
end

%% displaying message
waitbar(4 / 5, ...
    hWaitbar, ...
    'Saving processed localization data');

%% saving ungrouped and non-drift-corrected localization data (frame,xPos,yPos,locPrec,int)
writematrix(allData,fullfile(fPath,'SR_proc.txt'));

%% displaying message
waitbar(5 / 5, ...
    hWaitbar, ...
    'Rendering and saving processed SR image');

%% rendering grouped and drift-corrected SR image
imageSR_proc = zeros(round(size(imageRaw) .* magFactor));
allData(:,2 : 4) = allData(:,2 : 4) ./ imagePixelSize;
for particleId = 1 : size(allData,1)
    rows = round(allData(particleId,3)) - 5 : round(allData(particleId,3)) + 5;
    cols = round(allData(particleId,2)) - 5 : round(allData(particleId,2)) + 5;
    try
        for rowId = rows
            for colId = cols
                imageSR_proc(rowId,colId) = imageSR_proc(rowId,colId) + ...
                    allData(particleId,5) .* exp(-((colId - allData(particleId,2)) .^ 2 + (rowId - allData(particleId,3)) .^ 2) ./ ...
                    (2 * (allData(particleId,4) ^ 2)));
            end
        end
    catch
    end
end

%% saving ungrouped and non-drift-corrected image
imwrite(uint16(imageSR_proc),fullfile(fPath,'SR_proc.tif'));

%% displaying message
waitbar(5 / 5, ...
    hWaitbar, ...
    'Done');

%% --------- END OF CODE --------- %%
