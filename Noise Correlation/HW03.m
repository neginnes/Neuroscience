%  This script contains three parts:
%   1. Convert spike times to 1ms bins.
%   2. Remove bad/very low firing units.
%   3. Compute the trial-averaged population activity (PSTHs).
%
%  S struct is used to store the data:
%    S(igrat).trial(itrial).spikes  (num_units x num_1ms_timebins)
%    S(igrat).trial(itrial).counts  (num_units x num_20ms_timebins)
%    S(igrat).mean_FRs  (num_units x num_20ms_timebins)
%
%  Author: Ben Cowley, bcowley@cs.cmu.edu, Oct. 2016
%
%  Notes:
%    - automatically saves 'S' in ./spikes_gratings/


%% parameters

    SNR_threshold = 1.5;
    firing_rate_threshold = 1.0;  % 1.0 spikes/sec
    binWidth = 20;  % 20 ms bin width

    
%% parameters relevant to experiment

    length_of_gratings = 1;  % each gratings was shown for 1.28s, take the last 1s
    
    filenames{1} = './spikes_gratings/data_monkey1_gratings.mat';
    filenames{2} = './spikes_gratings/data_monkey2_gratings.mat';
    filenames{3} = './spikes_gratings/data_monkey3_gratings.mat';


    monkeys = {'monkey1', 'monkey2', 'monkey3'};
    

%%  spike times --> 1ms bins

    for imonkey = 1:length(monkeys)
        S = [];

        fprintf('binning spikes for %s\n', monkeys{imonkey});

        load(filenames{imonkey});
            % returns data.EVENTS

        num_neurons = size(data.EVENTS,1);
        num_gratings = size(data.EVENTS,2);
        num_trials = size(data.EVENTS,3);

        edges = 0.28:0.001:1.28;  % take 1ms bins from 0.28s to 1.28s

        for igrat = 1:num_gratings
            for itrial = 1:num_trials
                for ineuron = 1:num_neurons
                    S(igrat).trial(itrial).spikes(ineuron,:) = histc(data.EVENTS{ineuron, igrat, itrial}, edges);
                end
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(:,1:end-1);  % remove extraneous bin at the end
            end
        end

        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    




%%  Pre-processing:  Remove bad/very low firing units

    % remove units based on low SNR
    
    for imonkey = 1:length(monkeys)
        load(filenames{imonkey});
            % returns data.SNR
        keepNeurons = data.SNR >= SNR_threshold;
        keepNeuronsIdx = find(keepNeurons);
        clear data;
        
        fprintf('keeping units with SNRs >= %f for %s\n', SNR_threshold, monkeys{imonkey});
        
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons,:);
                S(igrat).trial(itrial).neuronNumber = keepNeuronsIdx;
            end
        end
        
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    
    % remove units with mean firing rates < 1.0 spikes/sec
    
    for imonkey = 1:length(monkeys)
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        FRs = [];   
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                FRs = [FRs sum(S(igrat).trial(itrial).spikes,2)/1.0];
            end
        end
        
        mean_FRs_gratings = mean(FRs,2);
        keepNeurons = mean_FRs_gratings >= firing_rate_threshold;
        keepNeuronsIdx = find(keepNeurons);
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons,:);
                S(igrat).trial(itrial).neuronNumber = keepNeuronsIdx;
            end
        end
           
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
        
    end


%%  Take spike counts in bins
    for imonkey = 1:length(monkeys)
        
        fprintf('spike counts in %dms bins for %s\n', binWidth, monkeys{imonkey});
        
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).counts = bin_spikes(S(igrat).trial(itrial).spikes, binWidth);
            end
        end
        
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    
    
%%  Compute trial-averaged population activity (PSTHs)

    for imonkey = 1:length(monkeys)
        fprintf('computing PSTHs for %s\n', monkeys{imonkey});
        
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            FRs = zeros(size(S(igrat).trial(1).counts));
            for itrial = 1:num_trials
                FRs = FRs + S(igrat).trial(itrial).counts;
            end
            S(igrat).mean_FRs = FRs / num_trials;
        end
        
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end


%% Part A

clear all
close all
clc

data1 = load ('pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\S_monkey1.mat');
data2 = load ('pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\S_monkey2.mat');
data3 = load ('pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\S_monkey3.mat');


degrees =  0 : 30 : 330;
nCnd = size(data1.S,2);

for cnd = 1 : nCnd
    totalFRs1(cnd,:) = sum(data1.S(cnd).mean_FRs,2);
    totalFRs2(cnd,:) = sum(data2.S(cnd).mean_FRs,2);
    totalFRs3(cnd,:) = sum(data3.S(cnd).mean_FRs,2);
end

activations1 = sum(totalFRs1);
activations2 = sum(totalFRs2);
activations3 = sum(totalFRs3);

mostActiveNeuron1 = find(activations1 == max(activations1));
mostActiveNeuron2 = find(activations2 == max(activations2));
mostActiveNeuron3 = find(activations3 == max(activations3));

figure()
set(gcf,'color','w');
plot(degrees,totalFRs1(:,mostActiveNeuron1));
hold on
plot(degrees,totalFRs2(:,mostActiveNeuron2));
hold on
plot(degrees,totalFRs3(:,mostActiveNeuron3));
hold on

title("Most active neuron's tuning curves for each monkey",'Interpreter','latex');
xlabel("Gratings' degrees",'Interpreter','latex');
ylabel('Average spikes','Interpreter','latex');
legend('Monkey1','Monkey2','Monkey3','interpreter','latex')

vq1 = interp1(degrees,totalFRs1(:,mostActiveNeuron1),0:360,'spline');
vq2 = interp1(degrees,totalFRs2(:,mostActiveNeuron2),0:360,'spline');
vq3 = interp1(degrees,totalFRs3(:,mostActiveNeuron3),0:360,'spline');

figure()
set(gcf,'color','w');
plot(0:360,vq1);
hold on
plot(0:360,vq2);
hold on
plot(0:360,vq3);
hold on
xlim([0,360])


title("Most active neuron's tuning curves for each monkey",'Interpreter','latex');
xlabel("Gratings' degrees",'Interpreter','latex');
ylabel('Average spikes','Interpreter','latex');
legend('Monkey1','Monkey2','Monkey3','interpreter','latex')
%%
figure()
set(gcf,'color','w');
subplot(3,1,1)
imagesc(1:size(totalFRs1,2),degrees,totalFRs1)
title("Monkey1",'Interpreter','latex');
ylabel("Gratings' degrees",'Interpreter','latex');
xlabel('Neurons','Interpreter','latex');
C = colorbar();
C.Label.String = 'Average Firing';

subplot(3,1,2)
imagesc(1:size(totalFRs2,2),degrees,totalFRs2)
title("Monkey2",'Interpreter','latex');
ylabel("Gratings' degrees",'Interpreter','latex');
xlabel('Neurons','Interpreter','latex');
C = colorbar();
C.Label.String = 'Average Firing';

subplot(3,1,3)
imagesc(1:size(totalFRs3,2),degrees,totalFRs3)
title("Monkey3",'Interpreter','latex'); 
ylabel("Gratings' degrees",'Interpreter','latex');
xlabel('Neurons','Interpreter','latex');
C = colorbar();
C.Label.String = 'Average Firing';

totalFRs = {};
totalFRs{1}= totalFRs1;
totalFRs{2}= totalFRs2;
totalFRs{3} = totalFRs3;
save('totalFRs','totalFRs');

%% Part B
clear all
close all
clc
degrees =  0 : 30 : 330;
coloredMap = zeros(3, 10, 10);
for imonkey = 1:3
    file = "pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\data_monkey" + num2str(imonkey) + "_gratings.mat";
    load(file)
    file = "pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\S_monkey" + num2str(imonkey) + ".mat";
    processedData = load(file);
    load('totalFRs.mat')
    FRs = totalFRs{imonkey};
    map = data.MAP;
    channels = data.CHANNELS;
    for i = 1:size(map,1)
        for j = 1:size(map,2)
            peak = 0;
            channelIndices = find(channels(:,1) == map(i,j));
            if ~isempty(channelIndices) && ~isempty(find(processedData.S(1).trial(1).neuronNumber == channelIndices(1)))
                neuron = find(processedData.S(1).trial(1).neuronNumber == channelIndices(1));
                peak = find(FRs(:,neuron) == max(FRs(:,neuron)));
            end 
            if peak
                coloredMap(imonkey, i, j) = degrees(peak(1));
            else
                coloredMap(imonkey, i, j) = 0;
            end
        end
    end
end

for imonkey = 1:3    
    file = "pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\data_monkey" + num2str(imonkey) + "_gratings.mat";
    load(file)
    map = data.MAP;
    channels = data.CHANNELS;
    figure()
    set(gcf,'color','w');
    image = squeeze(coloredMap(imonkey, :, :));
    imagesc(image);
    C = colorbar();
    C.Label.String = 'Orientation (degree)';
    title(['Pinwheel organization for monkey', num2str(imonkey)], 'Interpreter','latex');
end


%% Part C
clear all
close all
clc

load('totalFRs.mat')

for imonkey = 1:3
    FRs = totalFRs{imonkey};
    signalCorrelation = size(size(FRs,2));
    for i = 1 : size(FRs,2)
        for j = 1 : size(FRs,2)
            signalCorrelation(i,j) = corr(FRs(:,i),FRs(:,j));
        end
    end
    signalCorrelations{imonkey} = signalCorrelation;
end

for imonkey = 1 : 3
    file = "pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\S_monkey" + num2str(imonkey) + ".mat";
    processedData = load(file);
    nCnd = size(processedData.S,2);
    nTrials = size(processedData.S(1).trial,2);
    nNeurons = size(processedData.S(1).mean_FRs,1);
    neuronsVector = zeros(nNeurons,nTrials*nCnd);
    neuronVector = zeros(1,nTrials*nCnd);
    for neuron = 1 : nNeurons
        counter = 0;
        for cnd = 1 : nCnd
            for trialCount = 1 : nTrials
                counter = counter + 1;
                FRs = sum(processedData.S(cnd).trial(trialCount).spikes,2);
                neuronVector(counter) = FRs(neuron);
            end
            neuronVector(counter-nTrials +1 : counter) = zscore(neuronVector(counter-nTrials +1 : counter));
        end
        neuronsVector(neuron,:) = neuronVector;
    end
    noiseCorrelation = zeros(nNeurons);
    for i = 1 : nNeurons
        for j = 1 : nNeurons
            noiseCorrelation(i,j) = corr(neuronsVector(i,:)',neuronsVector(j,:)');
        end
    end
    noiseCorrelations{imonkey} = noiseCorrelation;

end

for imonkey = 1 : 3
    file = "pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\S_monkey" + num2str(imonkey) + ".mat";
    processedData = load(file);
    file = "pvc-11\data_and_scripts\data_and_scripts\spikes_gratings\data_monkey" + num2str(imonkey) + "_gratings.mat";
    load(file);

    nNeurons = size(processedData.S(1).mean_FRs,1);
    neuronsDistance = zeros(nNeurons);
    for i = 1 : nNeurons
        for j = 1 : nNeurons
            ineuronNumber = processedData.S(1).trial(1).neuronNumber(i);
            jneuronNumber = processedData.S(1).trial(1).neuronNumber(j);
            ielectrodeNumber = data.CHANNELS(ineuronNumber);
            if ~isempty(ielectrodeNumber)
                ielectrodeNumber = ielectrodeNumber(1);
            end
            jelectrodeNumber = data.CHANNELS(jneuronNumber);
            if ~isempty(jelectrodeNumber)
                jelectrodeNumber = jelectrodeNumber(1);
            end
            [px1,py1] = find(data.MAP == ielectrodeNumber);
            [px2,py2] = find(data.MAP == jelectrodeNumber);
            neuronsDistance(i,j) = 400*sqrt((px1-px2)^2 + (py1-py2)^2 );
        end
    end

    neuronsDistances{imonkey} = neuronsDistance;

end

%%
for imonkey = 1:3
    Matrix = neuronsDistances{imonkey};
    Matrix(triu(ones(size(Matrix)),1) == 0) = NaN;
    Matrix = reshape(Matrix,1,[]);
    NaNIndices = isnan(Matrix);
    Matrix = Matrix(~NaNIndices);
    neuronsDistances{imonkey} = Matrix;

    Matrix = noiseCorrelations{imonkey};
    Matrix(triu(ones(size(Matrix)),1) == 0) = NaN;
    Matrix = reshape(Matrix,1,[]);
    NaNIndices = isnan(Matrix);
    Matrix = Matrix(~NaNIndices);
    noiseCorrelations{imonkey} = Matrix;


    Matrix = signalCorrelations{imonkey};
    Matrix(triu(ones(size(Matrix)),1) == 0) = NaN;
    Matrix = reshape(Matrix,1,[]);
    NaNIndices = isnan(Matrix);
    Matrix = Matrix(~NaNIndices);
    signalCorrelations{imonkey} = Matrix;
end

%%  first figure

xcenters = 1000*(0 : 0.5 : 5);
wx = xcenters(2) - xcenters(1);
y1 = [];
y2 = [];
y3 = [];
y4 = [];
e1 = [];
e2 = [];
e3 = [];
e4 = [];
for imonkey = 1:3
    for i = 1 : length(xcenters)
        if i == length(xcenters)
            idx = find( (neuronsDistances{imonkey} >= xcenters(i) - wx/2) & signalCorrelations{imonkey} < -0.5);
            y1(i) = mean(noiseCorrelations{imonkey}(idx));
            e1(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & signalCorrelations{imonkey} >= -0.5 & signalCorrelations{imonkey} < 0);
            y2(i) = mean(noiseCorrelations{imonkey}(idx));
            e2(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & signalCorrelations{imonkey} >= 0 & signalCorrelations{imonkey} < 0.5);
            y3(i) = mean(noiseCorrelations{imonkey}(idx));
            e3(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & signalCorrelations{imonkey} >= 0.5);
            y4(i) = mean(noiseCorrelations{imonkey}(idx));
            e4(i) = var(noiseCorrelations{imonkey}(idx));
        else
            idx = find( (neuronsDistances{imonkey} >= xcenters(i) - wx/2) & signalCorrelations{imonkey} < -0.5);
            y1(i) = mean(noiseCorrelations{imonkey}(idx));
            e1(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & (neuronsDistances{imonkey} < xcenters(i) + wx/2) & signalCorrelations{imonkey} >= -0.5 & signalCorrelations{imonkey} < 0);
            y2(i) = mean(noiseCorrelations{imonkey}(idx));
            e2(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & (neuronsDistances{imonkey} < xcenters(i) + wx/2) & signalCorrelations{imonkey} >= 0 & signalCorrelations{imonkey} < 0.5);
            y3(i) = mean(noiseCorrelations{imonkey}(idx));
            e3(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & (neuronsDistances{imonkey} < xcenters(i) + wx/2) & signalCorrelations{imonkey} >= 0.5);
            y4(i) = mean(noiseCorrelations{imonkey}(idx));
            e4(i) = var(noiseCorrelations{imonkey}(idx));
        end
    end
    figure()
    set(gcf,'color','w');
    errorbar(0.001*xcenters,y1,e1);
    hold on
    errorbar(0.001*xcenters,y2,e2);
    hold on
    errorbar(0.001*xcenters,y3,e3);
    hold on
    errorbar(0.001*xcenters,y4,e4);
    hold on
    title("Monkey" + num2str(imonkey) ,'Interpreter','latex')
    xlabel('Distance between electrodes (mm)','Interpreter','latex');
    ylabel('Noise corelation','Interpreter','latex');
    legend('signal correlation $$<-0.5$$','signal correlation -0.5 to 0','signal correlation 0 to 0.5','signal correlation $$>0.5$$','interpreter','latex')
end

%% second figure
y1 = [];
y2 = [];
y3 = [];
y4 = [];
y5 =[];
e1 = [];
e2 = [];
e3 = [];
e4 = [];
e5 = [];

xcenters = -1 : 0.25: 1;

for imonkey = 1:3
    for i = 1 : length(xcenters)
        if i == 1 
            idx = find(signalCorrelations{imonkey} < xcenters(i) & neuronsDistances{imonkey} >= 0 & neuronsDistances{imonkey} < 1000);
            y1(i) = mean(noiseCorrelations{imonkey}(idx));
            e1(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & neuronsDistances{imonkey} >= 1000 & neuronsDistances{imonkey} < 2000);            
            y2(i) = mean(noiseCorrelations{imonkey}(idx));
            e2(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & neuronsDistances{imonkey} >= 2000 & neuronsDistances{imonkey} < 3000);
            y3(i) = mean(noiseCorrelations{imonkey}(idx));
            e3(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & neuronsDistances{imonkey} >= 3000 & neuronsDistances{imonkey} < 4000);
            y4(i) = mean(noiseCorrelations{imonkey}(idx));
            e4(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & neuronsDistances{imonkey} >= 4000 & neuronsDistances{imonkey} < 10000);
            y5(i) = mean(noiseCorrelations{imonkey}(idx));
            e5(i) = var(noiseCorrelations{imonkey}(idx));
        elseif i == length(xcenters)
            idx = find(signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 0 & neuronsDistances{imonkey} < 1000);
            y1(i) = mean(noiseCorrelations{imonkey}(idx));
            e1(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 1000 & neuronsDistances{imonkey} < 2000);            
            y2(i) = mean(noiseCorrelations{imonkey}(idx));
            e2(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 2000 & neuronsDistances{imonkey} < 3000);
            y3(i) = mean(noiseCorrelations{imonkey}(idx));
            e3(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 3000 & neuronsDistances{imonkey} < 4000);
            y4(i) = mean(noiseCorrelations{imonkey}(idx));
            e4(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 4000 & neuronsDistances{imonkey} < 10000);
            y5(i) = mean(noiseCorrelations{imonkey}(idx));
            e5(i) = var(noiseCorrelations{imonkey}(idx));
        else
            idx = find(signalCorrelations{imonkey} < xcenters(i) & signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 0 & neuronsDistances{imonkey} < 1);
            y1(i) = mean(noiseCorrelations{imonkey}(idx));
            e1(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 1000 & neuronsDistances{imonkey} < 2000);            
            y2(i) = mean(noiseCorrelations{imonkey}(idx));
            e2(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 2000 & neuronsDistances{imonkey} < 3000);
            y3(i) = mean(noiseCorrelations{imonkey}(idx));
            e3(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 3000 & neuronsDistances{imonkey} < 4000);
            y4(i) = mean(noiseCorrelations{imonkey}(idx));
            e4(i) = var(noiseCorrelations{imonkey}(idx));
            idx = find(signalCorrelations{imonkey} < xcenters(i) & signalCorrelations{imonkey} >= xcenters(i-1) & neuronsDistances{imonkey} >= 4000 & neuronsDistances{imonkey} < 10000);
            y5(i) = mean(noiseCorrelations{imonkey}(idx));
            e5(i) = var(noiseCorrelations{imonkey}(idx));
        end
    end
    figure()
    set(gcf,'color','w');
    errorbar(xcenters,y1,e1);
    hold on
    errorbar(xcenters,y2,e2);
    hold on
    errorbar(xcenters,y3,e3);
    hold on
    errorbar(xcenters,y4,e4);
    hold on
    errorbar(xcenters,y5,e5);
    hold on 
    title("Monkey" + num2str(imonkey) ,'Interpreter','latex');
    xlabel('Signal correlation','Interpreter','latex');
    ylabel('Noise corelation','Interpreter','latex');
    legend('Distance 0-1 mm','Distance 1-2 mm','Distance 2-3 mm','Distance 3-4 mm','Distance 4-10 mm','interpreter','latex');
end

%% third figure

xcenters = 1000*(0.5:0.5:5);
wx = xcenters(2)-xcenters(1);
ycenters = -1 : 0.2: 1;
wy = ycenters(2)-ycenters(1);
y = [];
color = [];

for imonkey = 1:3
    for i = 1 : length(xcenters) 
        for j = 1 : length(ycenters) 
            idx = find((neuronsDistances{imonkey} >= xcenters(i) - wx/2) & (neuronsDistances{imonkey} < xcenters(i) + wx/2) & ...
                (signalCorrelations{imonkey} >= ycenters(j) - wy/2) & (signalCorrelations{imonkey} < ycenters(j) + wy/2));
            color(i,j) = mean(noiseCorrelations{imonkey}(idx));
        end
    end
    figure()
    set(gcf,'color','w');
    pcolor(0.001*xcenters,ycenters,color');
    hold on
    C = colorbar();
    C.Label.String = 'Noise correlation';
    title("Monkey" + num2str(imonkey) ,'Interpreter','latex')
    xlabel('Distance between electrodes (mm)','Interpreter','latex');
    ylabel('Signal corelation','Interpreter','latex');
end


















