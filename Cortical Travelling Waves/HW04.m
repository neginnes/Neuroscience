clear all
close all
clc

% data cleaning
Data = load('ArrayData\ArrayData.mat');
cleanTrials = load('ArrayData\CleanTrials.mat');

for i = 1 : length(Data.chan)
    Data.chan(i).lfp = Data.chan(i).lfp(:,cleanTrials.Intersect_Clean_Trials);
end
save('cleanedData.mat','Data')

%% Part 1.A
clear all
close all
clc

load('cleanedData.mat');
nChannels = length(Data.chan);
nTrials = size(Data.chan(1).lfp,2);

% data denoising 
dt = abs(Data.Time(2) - Data.Time(1)); % Sampling period 
Fs = 1/dt;                             % Sampling frequency                         
L = length(Data.Time);                 % Length of signal
f = Fs*(0:(L/2))/L;
f(1) = [];

ri = randi(nChannels);
rj = randi(nTrials);
for i = 1 : nChannels
    Y = fft(Data.chan(i).lfp);
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1,:);
    P1(2:end-1,:) = 2*P1(2:end-1,:);
    P1(1,:) = [];
    y = log10(P1);
    x = log10(f');
    for j = 1 : nTrials
        fitobject = fit(x,y(:,j),'poly1');
        a = fitobject.p1;
        b = fitobject.p2;
        noise =  a.*x + b;
        if (i == ri) && (j == rj)
            figure()
            set(gcf,'color','w');
            plot(x,y(:,j))
            hold on
            plot(x,noise)
            title('A randomly selected trial','Interpreter','latex')
            xlabel('Logarithmic Frequancy (Hz)','Interpreter','latex')
            legend('Logarithmic LFP Furier Transform Magnitude', 'Logarithmic Color Noise' ,'Interpreter','latex')
            hold off
        end
        Data.chan(i).cleanLFP(:,j) = y(:,j) - noise;
    end
end
%%
dominantFreqs = [];
figure()
for i = 1 : nChannels
    AVG_Power_spectogram = mean(exp(Data.chan(i).cleanLFP).^2,2);
    M = max(AVG_Power_spectogram);
    idx = find(AVG_Power_spectogram == M);
    dominantFreqs = [dominantFreqs, f(idx(1))];
    set(gcf,'color','w');
    subplot(8,6,i)
    plot(f,AVG_Power_spectogram)
    title("channel " + num2str(i) ,'Interpreter','latex')
    xlabel('Frequancy(Hz)','Interpreter','latex')
    ylabel('Avg PS' ,'Interpreter','latex')
end


figure()
set(gcf,'color','w');
plot(dominantFreqs)
title('Most Dominant Frequancies','Interpreter','latex')
xlabel('Channel','Interpreter','latex')
ylabel('Frequancy(Hz)' ,'Interpreter','latex')

arraySize = size(Data.ChannelPosition);
colors = zeros(arraySize);
for i = 1 : arraySize(1)*arraySize(2)
    idx = find(Data.ChannelPosition == i);
    if (~isempty(idx))
        colors(idx(1)) = dominantFreqs(i);
    end
end
figure()
set(gcf,'color','w');
imagesc(1:arraySize(2),1:arraySize(1),colors)
colorbar();
title('Most Dominant Frequancies','Interpreter','latex')
xlabel('Array','Interpreter','latex')

%% Part 1.B
close all
clc

k = 2;
[indices,C] = kmeans(dominantFreqs',k);
for i = 1 : arraySize(1)*arraySize(2)
    idx = find(Data.ChannelPosition == i);
    if (~isempty(idx))
        colors(idx(1)) = indices(i);
    end
end
dominantFreq = C(mode(indices));
save('dominantFreq.mat','dominantFreq');
figure()
set(gcf,'color','w');
imagesc(1:arraySize(2),1:arraySize(1),colors)
colorbar();
title('Clustering The Most Dominant Frequancies','Interpreter','latex')
xlabel('Array','Interpreter','latex')


for i = 1 : arraySize(1)
    for j = 1 : arraySize(2)
        if (~isnan(Data.ChannelPosition(i,j)))
            text(j-0.25,i,["center = " , num2str(C(indices(Data.ChannelPosition(i,j))))],'interpreter','latex');
            hold on
        end
    end
end


%% Part 1.C
clear all
close all
clc

load('cleanedData.mat');
nChannels = length(Data.chan);
nTrials = size(Data.chan(1).lfp,2);

% data denoising 
dt = abs(Data.Time(2) - Data.Time(1)); % Sampling period 
Fs = 1/dt;                             % Sampling frequency                         


for i = 1 : nChannels
    [s,f,t] = stft(Data.chan(i).lfp,Fs,'Window',hann(64,'periodic'));
    t = t - 1.2;
    s = abs(s(f>0,:,:));
    f = f(f>0);
    y = log10(s);
    x = log10(f);
    for j = 1 : nTrials
        for k = 1 : size(s,2)
            fitobject = fit(x,y(:,k,j),'poly1');
            a = fitobject.p1;
            b = fitobject.p2;
            noise =  a.*x + b;
            Data.chan(i).lfpSTFT(:,k,j) = y(:,k,j) - noise;
        end
    end
end
%%
channelSpectogram = zeros(nChannels,size(s,1),size(s,2));
AvgSpectogram = 0;
for i = 1 : nChannels
    AvgSpectogram = AvgSpectogram + mean((10.^(Data.chan(i).lfpSTFT)).^2,3);
    channelSpectogram(i,:,:) =  mean((10.^(Data.chan(i).lfpSTFT)).^2,3);
end
AvgSpectogram = AvgSpectogram / nChannels;

[idx1,idx2] = find(AvgSpectogram == max(AvgSpectogram));
dmtFreq =  f(idx1(1));
figure()
set(gcf,'color','w');
imagesc(t,f,AvgSpectogram)
colorbar();
title("Average Power Spectrum of LFP Signals Through Time (Dominant frequancy = " + num2str(dmtFreq) + " Hz)",'Interpreter','latex')
xlabel('t(s)','Interpreter','latex')
ylabel('f(Hz)','Interpreter','latex')


r = randi(nChannels);
[idx1,idx2] = find(squeeze(channelSpectogram(r,:,:)) == max(squeeze(channelSpectogram(r,:,:))));
dmtFreq =  f(idx1(1));
figure()
set(gcf,'color','w');
imagesc(t,f,squeeze(channelSpectogram(r,:,:)))
colorbar();
title("Channel number "+num2str(r) +" Power Spectrum of LFP Signals Through Time (Dominant frequancy = " + num2str(dmtFreq) + " Hz)",'Interpreter','latex')
xlabel('t(s)','Interpreter','latex')
ylabel('f(Hz)','Interpreter','latex')

%% Part 2.A
clear all
close all
clc
load('cleanedData.mat');
load('dominantFreq.mat');
nChannels = length(Data.chan);
nTrials = size(Data.chan(1).lfp,2);
% data denoising 
dt = abs(Data.Time(2) - Data.Time(1)); % Sampling period 
Fs = 1/dt;                             % Sampling frequency                         
k = 2;
Fc = dominantFreq;
[b,a] = butter(k/2,[Fc-0.5, Fc+0.5]/(Fs/2),'bandpass'); 

for i = 1 : nChannels
    y = filter(b,a,Data.chan(i).lfp);
    Data.chan(i).filteredLfp = y;
end

%% Part 2.B
close all
clc
arraySize = size(Data.ChannelPosition);
nTimePoints = size(Data.chan(1).lfp,1);
Phi_xyt =  zeros(arraySize(1),arraySize(2),nTrials,nTimePoints);

for i = 1 : nChannels
    [xpos,ypos] = find(Data.ChannelPosition == i);
    Phi_xyt(xpos,ypos,:,:) = (angle(1i *(hilbert(Data.chan(i).filteredLfp)) + Data.chan(i).filteredLfp))';
end

%% Part 2.C
close all
clc

writerObj = VideoWriter("demo");
writerObj.FrameRate = 20;
open(writerObj);

%r = randi(nTrials);
r = 10;
fig = figure();
for j = 1 : nTimePoints
    imagesc(squeeze(cos(Phi_xyt(:,:,r,j))));
    colormap('hot')
    frame = getframe(fig);
    for i = 1 : 5
        writeVideo(writerObj,frame);
    end
end

close(writerObj)      

%% Part 2.D
close all
clc

%r = randi(nTrials);
r = 10;
distance = 400/(10^6);
Phi_t = squeeze(Phi_xyt(:,:,r,:));
Phi_t2 = Phi_t;
Phi_t2(:,:,end+1) =  zeros(size(Phi_t,1),size(Phi_t,2));
deriv_Phi_t = Phi_t2(:,:,2:end) - Phi_t2(:,:,1:end-1);
[grad_Phix,grad_Phiy] = gradient(Phi_t);

direction = -squeeze(atan2d(mean(mean(grad_Phiy)),mean(mean(grad_Phix))));
speed = squeeze(Fs*abs(mean(mean(deriv_Phi_t)))./mean(mean(sqrt(  (grad_Phix./distance).^2 +  (grad_Phiy./distance).^2  ))));
PGD = squeeze(sqrt( (mean(mean(grad_Phix))./distance).^2 + (mean(mean(grad_Phiy))./distance).^2 )./mean(mean(sqrt( (grad_Phix./distance).^2 +  (grad_Phiy./distance).^2  ))));

%% Part 2.E
close all
clc

writerObj = VideoWriter("demo");
writerObj.FrameRate = 20;
open(writerObj);

%r = randi(nTrials);
r = 10;
distance = 400/(10^6);
Phi_t = squeeze(Phi_xyt(:,:,r,:));
Phi_t2 = Phi_t;
Phi_t2(:,:,end+1) =  zeros(size(Phi_t,1),size(Phi_t,2));
deriv_Phi_t = Phi_t2(:,:,2:end) - Phi_t2(:,:,1:end-1);
[grad_Phix,grad_Phiy] = gradient(Phi_t);
direction = -squeeze(atan2d(mean(mean(grad_Phiy)),mean(mean(grad_Phix))));
speed = squeeze(Fs*abs(mean(mean(deriv_Phi_t)))./mean(mean(sqrt(  (grad_Phix./distance).^2 +  (grad_Phiy./distance).^2  ))));
PGD = squeeze(sqrt( (mean(mean(grad_Phix))./distance).^2 +  (mean(mean(grad_Phiy))./distance).^2 )./mean(mean(sqrt(  (grad_Phix./distance).^2 +  (grad_Phiy./distance).^2  ))));

fig = figure();
for j = 1 : nTimePoints
    imagesc(squeeze(cos(Phi_xyt(:,:,r,j))));
    title("PGD = " + num2str(PGD(j)) + ", speed = " + num2str(100*speed(j)) + " cm/s" + ", direction = " + num2str(direction(j)) + " degree" ,'interpreter','latex')
    xlabel('Array','interpreter','latex')
    text(arraySize(2)/2,arraySize(1)/2,"t = " + num2str(Data.Time(j)) + " s",'interpreter','latex')
    colormap('hot')
    frame = getframe(fig);
    for i = 1 : 5
        writeVideo(writerObj,frame);
    end
end

close(writerObj)   

%% Part F
close all
clc 
direction = [];
speed = [];
PGD = [];
for trialCount = 1 : nTrials
    distance = 400/(10^6);
    Phi_t = squeeze(Phi_xyt(:,:,trialCount,:));
    Phi_t2 = Phi_t;
    Phi_t2(:,:,end+1) =  zeros(size(Phi_t,1),size(Phi_t,2));
    deriv_Phi_t = Phi_t2(:,:,2:end) - Phi_t2(:,:,1:end-1);
    [grad_Phix,grad_Phiy] = gradient(Phi_t);
    direction = [direction;-squeeze(atan2d(mean(mean(grad_Phiy)),mean(mean(grad_Phix))))];
    speed = [speed;squeeze(Fs*abs(mean(mean(deriv_Phi_t)))./mean(mean(sqrt(  (grad_Phix./distance).^2 +  (grad_Phiy./distance).^2  ))))];
    PGD = [PGD;squeeze(sqrt( (mean(mean(grad_Phix))./distance).^2 + (mean(mean(grad_Phiy))./distance).^2 )./mean(mean(sqrt( (grad_Phix./distance).^2 +  (grad_Phiy./distance).^2  ))))];
end
figure()
set(gcf,'color','w');
histogram(direction,150)
title('Histogram of propagation direction through all trials and times','interpreter','latex')
xlabel('direction(degree)','interpreter','latex')
xlim([-180,180])

figure()
set(gcf,'color','w');
histogram(speed*100)
title('Histogram of speed through all trials and times','interpreter','latex')
xlabel('speed(cm/s)','interpreter','latex')
xlim([0,80])

figure()
set(gcf,'color','w');
histogram(PGD)
title('Histogram of PGD through all trials and times','interpreter','latex')
xlabel('PGD','interpreter','latex')

%% Part G
close all
clc 
Average_Speed = mean(speed(PGD>0.3))*100;
