%% Part1-A
clear all
close all
clc

lambda = 100;
tSim = 1;
nTrials = 20;
[spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials);
Title = 'Raster plot of 20 trials with poisson process, time interval = 1s, rate = 100, dt = 1ms';
plotRaster(spikeMat, tVec, Title)

%% Part1-B
clear all
close all
clc

lambda = 100;
tSim = 1;
nTrials = 1000;
[spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials); 
spike_count = sum(spikeMat,1);

histogram(spike_count,200)
title('calculated spike count probability histogram', 'interpreter','latex')
xlabel('N','interpreter','latex')

figure()
histogram(spike_count,200)
hold on
histogram(poissrnd(lambda,1,length(spike_count)),100)
xlabel('N','interpreter','latex')
legend('calculated spike count probability histogram','theoretical (Poisson) spike count density','interpreter','latex')

%% Part1-C
clear all
close all
clc 

lambda = 100;
tSim = 1;
nTrials = 1000;
[spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials); 
spike_intervals = [];
for trialCount = 1:size(spikeMat,1)
    spikePos = tVec(spikeMat(trialCount, :));
    spike_intervals = [spike_intervals, diff(spikePos)];
end

h = histogram(spike_intervals,200);
Xtext_idx = h.BinEdges(round(end/3));
Ytext_idx = round(0.5*max(h.Values));
xlabel('$$\tau(s)$$','interpreter','latex')
text(Xtext_idx,Ytext_idx,['mean = ',num2str(mean(spike_intervals)),'     ,      std = ',num2str(std(spike_intervals))],'interpreter','latex');
title('calculated Inter-spike interval (ISI) histogram','interpreter','latex')
xlabel('$$\tau(s)$$','interpreter','latex')

figure()
histogram(spike_intervals,200);
hold on
histogram(exprnd(1/lambda,1,length(spike_intervals)),200);
legend('calculated Inter-spike interval (ISI) histogram','theoretical (exponential) inters-pike interval density','interpreter','latex')
xlabel('$$\tau(s)$$','interpreter','latex')

%% Part1-D-1

clear all
close all
clc

lambda = 100;
tSim = 1;
nTrials = 20;
[spikeMat1, tVec] = poissonSpikeGen(lambda, tSim, nTrials);
k = 5;
spikeMat2 = Kth_Spike_RenewalSpikeGen(spikeMat1, k);

subplot(1,2,1)
Title = 'Raster plot of 20 trials, time interval = 1s, rate = 100, dt = 1ms - first method';
plotRaster(spikeMat1, tVec, Title)
subplot(1,2,2)
Title = 'Raster plot of 20 trials, time interval = 1s, rate = 100, dt = 1ms - second method';
plotRaster(spikeMat2, tVec, Title)


%% Part1-D-2
clear all
close all
clc

lambda = 100;
tSim = 1;
nTrials = 1000;
[spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials); 
k = 5;
spikeMat = Kth_Spike_RenewalSpikeGen(spikeMat, k);
spike_count = sum(spikeMat,1);

histogram(spike_count,200)
title('calculated spike count probability histogram', 'interpreter','latex')
xlabel('N','interpreter','latex')

figure()
histogram(spike_count,200)
hold on
histogram(poissrnd(lambda/k,1,length(spike_count)),100)
xlabel('N','interpreter','latex')
legend('calculated spike count probability histogram','theoretical (Poisson) spike count density with rate = lambda/k = 20','interpreter','latex')


%% Part1-D-3
clear all
close all
clc 

lambda = 100;
tSim = 1;
nTrials = 1000;
[spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials); 
k = 5;
spikeMat = Kth_Spike_RenewalSpikeGen(spikeMat, k);
spike_intervals = [];
for trialCount = 1:size(spikeMat,1)
    spikePos = tVec(spikeMat(trialCount, :));
    spike_intervals = [spike_intervals, diff(spikePos)];
end


h = histogram(spike_intervals,200);
Xtext_idx = h.BinEdges(round(end/3));
Ytext_idx = round(0.5*max(h.Values));
xlabel('$$\tau(s)$$','interpreter','latex')
text(Xtext_idx,Ytext_idx,['mean = ',num2str(mean(spike_intervals)),'     ,      std = ',num2str(std(spike_intervals))],'interpreter','latex');
title('calculated Inter-spike interval (ISI) histogram','interpreter','latex')
xlabel('$$\tau(s)$$','interpreter','latex')

figure()
histogram(spike_intervals,200)
hold on
histogram(exprnd(1/(lambda/k),1,length(spike_intervals)),100)
xlabel('$$\tau(s)$$','interpreter','latex')
legend('calculated Inter-spike interval (ISI) histogram','theoretical (exponential) inters-pike interval density with rate = lambda/k = 20','interpreter','latex')

%% Part1-D
clear all
close all
clc

lambda = 100;
tSim = 1;
nTrials = 100;
[spikeMat1, tVec] = poissonSpikeGen(lambda, tSim, nTrials); 
K = [1,2,3,4,5,10,15,25,50];
CVs = [];
i = 1;
for k = K
    subplot(3,3,i)
    spikeMat2 = Kth_Spike_RenewalSpikeGen(spikeMat1, k);
    CV = CV_Calculator(spikeMat2,tVec);
    plot(1:nTrials,CV)
    title(['Cv for different spike trains and k = ', num2str(k)],'interpreter','latex')
    xlabel('Trial number','interpreter','latex')
    ylabel("Cv",'interpreter','latex')
    hold on
    plot(1:nTrials,ones(1,nTrials)*mean(CV))
    legend('Cv','AVG of Cv','interpreter','latex')
    i = i + 1;
end

%% Part1-G
clear all
close all
clc


tSim = 5;
nTrials = 100;
Tau_bar = 0.0001:0.0001:0.03;
Lambda = 1./Tau_bar;
RefPeriod = [0,0.001];
K = [1,4,21];
CV = [];
c = 0;
for refPeriod = RefPeriod
    for k = K
        c = c + 1;
        Cv = [];
        for lambda = Lambda
            CV = [];
            for trialCount = 1:nTrials
                    SpikeTimes1 = refPoissonSpikeGen(lambda, tSim, refPeriod);
                    SpikeTimes2 = downsample(SpikeTimes1,k);
                    spike_intervals =  diff(SpikeTimes2);
                if (length(spike_intervals) ~= 0)
                    CV = [CV,std(spike_intervals)/(mean(spike_intervals))];
                else
                    CV = [CV,0];
                end
            end
            Cv = [Cv,mean(CV)];
        end
        plot(Tau_bar,Cv)
        hold on
%         idx = round(c*length(Tau_bar)/(length(K)*length(RefPeriod)));
%         text(Tau_bar(idx),1.5*CV(idx),string(['k = ',num2str(k),', Tref = ',num2str(refPeriod)]),'interpreter','latex')
    end 
end
title('Cv across different firing rates and different refractory periods','interpreter','latex')
xlabel('$$\bar{{\Delta\tau}}$$','interpreter','latex')
ylabel('Cv','interpreter','latex')
legend('k = 1  ,  Tref = 0','k = 4 ,  Tref = 0','k = 21, Tref = 0','k = 1 ,  Tref = 1ms','k = 4 ,  Tref = 1ms','k = 21,  Tref = 1ms','interpreter','latex')

%% Part2-A
clear all
close all
clc

tSim = 0.1;
Tau = 0.01;
RI = 20/1000;
Vth = 15/1000;
dt = 1/1000;
refPeriod = 0.005;
cnt = 1;
V = zeros(1,length(0 : dt : tSim-dt));
spike_flage = 0;
t = 0;
while (t < tSim-dt)
    if (spike_flage ==1)
        V(cnt) = 0;
    else    
        if (cnt~=1)
            dV = (-V(cnt-1) + RI)/Tau;
            V(cnt) = V(cnt-1)+ dV*dt; 
        end
    end
    if (V(cnt) >= Vth)
        V(cnt) = 30/1000;
        spike_flage = 1;
        t = t + refPeriod;
        if (cnt + refPeriod/dt   < length(V))
            cnt = cnt + refPeriod/dt;
        else
            t = tSim-dt;
        end
    else
        spike_flage = 0;
    end
    cnt = cnt + 1;
    t = t + dt;
end

t = 0 : dt : tSim-dt;
subplot(2,1,1)
plot(t,V)
title('The time-course of the membrane potential','interpreter','latex')
xlabel('t(s)','interpreter','latex')
ylabel('V(t)','interpreter','latex')

subplot(2,1,2)
plot(t,RI*ones(1,length(t)))
title('I(t)','interpreter','latex')
xlabel('t(s)','interpreter','latex')
ylabel('I(t)','interpreter','latex')
ylim([-0.03,0.03])

%% Part2-C
clear all
close all
clc

lambda = 100;
tSim = 0.1;
tpeak = 0.004;
Tau = 0.01;
R = 1/1000;
Vth = 15/1000;
alpha = 20*1000;
refPeriod = 0.005;
plt_flag = 1;


V = LIF_VariableCurrent(lambda, tSim, tpeak, alpha, Tau, R, Vth, refPeriod, plt_flag);

%% Part2-C-additional plots explaining the effect of width and magnitude of EPSCs on Cv
clear all
close all
clc

lambda = 100;
tSim = 0.1;
Tau = 0.01;
R = 1/1000;
Vth = 15/1000;
alpha = 20*1000;
plt_flag = 0;
refPeriod = 0.005;
dt = 1/1000;

tpeaks = [0.0001,0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.05];
i = 1;
for tpeak = tpeaks
    V = LIF_VariableCurrent(lambda, tSim, tpeak, alpha, Tau, R, Vth, refPeriod, plt_flag);
    t = 0 : dt : tSim-dt;
    subplot(3,3,i)
    plot(t,V)
    Title = ['EPSC width =  ',num2str(1000*tpeak),'ms in 0.1s simulation'];
    title(Title,'interpreter','latex')
    xlabel('t(s)','interpreter','latex')
    ylabel('V(t)','interpreter','latex')
    i = i + 1;
end




tpeak = 0.004;
alphas = [0.1,1,10,100,1000,10000,20000,50000,100000];
i = 1;
figure()
for alpha = alphas
    V = LIF_VariableCurrent(lambda, tSim, tpeak, alpha, Tau, R, Vth, refPeriod, plt_flag);
    t = 0 : dt : tSim-dt;
    subplot(3,3,i)
    plot(t,V)
    Title = ['EPSC magnitude =  ',num2str(alpha)];
    title(Title,'interpreter','latex')
    xlabel('t(s)','interpreter','latex')
    ylabel('V(t)','interpreter','latex')
    i = i + 1;
end


%% Part2-C- genarating contour plot
clear
close all
clc

lambda = 200; tSim = 100; tpeak = 1; R = 1/1000; Vth = 15/1000; alpha = 20*1000; Vm = 30/1000; dt = 1/1000; refPeriod = 0.005;
Tau = 1000*(0.1/1000:0.1/1000:10/1000);
K = 1:100;
CV = zeros(length(Tau),length(K));
row = 1;
for tau = Tau
    col = 1;
    for k = K
        nTrials = 1;
        [spikeMat1, tVec] = poissonSpikeGen(lambda, tSim, nTrials);
        spikeMat2 = Kth_Spike_RenewalSpikeGen(spikeMat1, k);

        t = 0: dt :tSim-dt;
        Is = t.*(exp(-t/tpeak));
        I = alpha*conv(spikeMat2,Is);
        I = I(1:length(t));
        V = zeros(1,length(0 : dt : tSim-dt));
        spike_flage = 0;
        flag = 0;
        t = dt;
        cnt = 2;
        while (t < tSim-dt)
            if (spike_flage ==1)
                V(cnt) = 0;
            else
                dV = dt*(-V(cnt-1) + R*I(cnt-1))/tau;
%                 ghabli = V(cnt-1)
%                 spike_flage
                V(cnt) = V(cnt-1)+ dV;
%                 V(cnt)
%                 pause(5)
            end
            if (V(cnt) >= Vth)
%                 'yes'
%                 pause(5)
                V(cnt) = 30/1000;
                spike_flage = 1;
                t = t + refPeriod;
                if (cnt + refPeriod/dt   < length(V))
                    cnt = cnt + refPeriod/dt;
                else
                    t = tSim-dt;
                end
            else
                spike_flage = 0;
            end
            cnt = cnt + 1;
            t = t + dt;
        end

            
%         subplot(2,1,1)
%         plot(V)
%         subplot(2,1,2)
%         plot(I(1:1000))
%         pause(1)
        spikeMat = (V == Vm);
%                     subplot(6,1,1)
%             stem(spikeMat1)
%             subplot(6,1,2)
%             stem(spikeMat2)
%             subplot(6,1,3)
%             plot(Is)
%             subplot(6,1,4)
%             plot(I)
%             subplot(6,1,5)
%             plot(V)
%             subplot(6,1,6)
%             stem(spikeMat)
%             pause(1)
        spike_intervals =  diff(tVec(spikeMat==1)) ;
        if (length(spike_intervals) ~= 0)
            CV(row,col) = std(spike_intervals)/(mean(spike_intervals));
        else
            CV(row,col) = 0;
        end
        col = col + 1;
    end
    row = row + 1;
end
contourf(Tau,K,CV')
set(gca,'xscale','log');
set(gca,'yscale','log');
title('CV contors','interpreter','latex')
xlabel('$$\tau(ms)$$','interpreter','latex')
ylabel('k','interpreter','latex')

%% Part2-D
clear
close all
clc

lambda = 100;
tSim = 0.1;
tpeak = 0.001;
Tau = 0.01;
R = 1/1000;
Vth = 15/1000;
alpha = 20*1000;
refPeriod = 0.005;

Percentage_of_inhibitory_neurons = [10,30,45,50,70];
for p = Percentage_of_inhibitory_neurons
    dt = 1/1000;
    nTrials = 100;
    [spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials);
    randomVector = rand(1,nTrials)<= p/100;
    idx = find(randomVector == 1);
    Sign = ones(nTrials,1);
    Sign(idx) = -1;
    Sign = Sign*ones(1,size(spikeMat,2));
    spikeMat = spikeMat.*Sign;
    t = 0: dt :tSim-dt;
    Is = t.*(exp(-t/tpeak));
    I = 0;
    for trialCount = 1 : nTrials
        spikeTrain = spikeMat(trialCount,:);
        I = I + alpha*conv(spikeTrain,Is);
    end
    cnt = 1;
    V = zeros(1,length(0 : dt : tSim-dt));
    spike_flage = 0;
    t = 0;
    while(t < tSim-dt)
        if (spike_flage ==1)
            V(cnt) = 0;
        else    
            if (cnt~=1)
                dV = (-V(cnt-1) + R*I(cnt-1))/Tau;
                V(cnt) = V(cnt-1)+ dV*dt; 
            end
        end
        if (V(cnt) >= Vth)
            V(cnt) = 30/1000;
            spike_flage = 1;
            t = t + refPeriod;
            if (cnt + refPeriod/dt   < length(V))
                cnt = cnt + refPeriod/dt;
            else
                t = tSim-dt;
            end
        else
            spike_flage = 0;
        end
        cnt = cnt + 1;
        t = t + dt;
    end
    figure()
    t = 0 : dt : tSim-dt;
    subplot(2,1,1)
    plot(t,V)
    title('The time-course of the membrane potential by a time-varying input current I(t)','interpreter','latex')
    xlabel('t(s)','interpreter','latex')
    ylabel('V(t)','interpreter','latex')

    subplot(2,1,2)
    plot(t,I(1:length(t)))
    title('I(t)','interpreter','latex')
    xlabel('t(s)','interpreter','latex')
    ylabel('I(t)','interpreter','latex')

end



%% Part2-D-Cv
clear
close all
clc

lambda = 100;
tSim = 0.1;
tpeak = 0.001;
Tau = 0.01;
R = 1/1000;
Vth = 15/1000;
Vm = 30/1000;
alpha = 20*1000;
refPeriod = 0.005;

Percentage_of_inhibitory_neurons = 0:0.1:100;
CV = zeros(size(Percentage_of_inhibitory_neurons));
c = 1;
for p = Percentage_of_inhibitory_neurons
    dt = 1/1000;
    nTrials = 1000;
    [spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials);
    randomVector = rand(1,nTrials)<= p/100;
    idx = find(randomVector == 1);
    Sign = ones(nTrials,1);
    Sign(idx) = -1;
    Sign = Sign*ones(1,size(spikeMat,2));
    spikeMat = spikeMat.*Sign;
    t = 0: dt :tSim-dt;
    Is = t.*(exp(-t/tpeak));
    I = 0;
    for trialCount = 1 : nTrials
        spikeTrain = spikeMat(trialCount,:);
        I = I + alpha*conv(spikeTrain,Is);
    end
    cnt = 1;
    V = zeros(1,length(0 : dt : tSim-dt));
    spike_flage = 0;
    t = 0;
    while(t < tSim-dt)
        if (spike_flage ==1)
            V(cnt) = 0;
        else    
            if (cnt~=1)
                dV = (-V(cnt-1) + R*I(cnt-1))/Tau;
                V(cnt) = V(cnt-1)+ dV*dt; 
            end
        end
        if (V(cnt) >= Vth)
            V(cnt) = 30/1000;
            spike_flage = 1;
            t = t + refPeriod;
            if (cnt + refPeriod/dt   < length(V))
                cnt = cnt + refPeriod/dt;
            else
                t = tSim-dt;
            end
        else
            spike_flage = 0;
        end
        cnt = cnt + 1;
        t = t + dt;
    end
    spikeMat = (V == Vm);
    spike_intervals =  diff(tVec(spikeMat==1)) ;
    if (~isempty(spike_intervals))
        CV(c) = std(spike_intervals)/(mean(spike_intervals));
    else
        CV(c) = 0;
    end
    c = c + 1;
end

plot(Percentage_of_inhibitory_neurons,CV)
title('dependency of Cv on the percentage of inhibtory neuons','interpreter','latex')
xlabel('ratio','interpreter','latex')
ylabel('CV','interpreter','latex')

%% Part2-E
clear
close all
clc

lambda = 200;
tSim = 0.5;
dt = 1/1000;
tVec = 0:dt:tSim-dt;
Ratios = 0.01*(1:100);
Ds = (dt:dt:tSim-dt);
refPeriod = 0.001;
spikeTimes1 = refPoissonSpikeGen(lambda, tSim, refPeriod);
spikeTimes1= dt*round(spikeTimes1/dt);
spikeTrain1 = zeros(size(tVec));
for i = 1 : length(spikeTimes1)
    idx = find(tVec == spikeTimes1(i));
    if (length(idx)~=0)
        spikeTrain1(idx) = 1;
    end
end
row = 1;
for D = Ds
    col = 1;
    for Ratio = Ratios
        spikeTrain2 = CoincidenceDitector(spikeTrain1,Ratio,D);
        spike_intervals =  diff(tVec(spikeTrain2==1)) ;
        if (length(spike_intervals) ~= 0)
            CV(row,col) = std(spike_intervals)/(mean(spike_intervals));
        else
            CV(row,col) = 0;
        end
        col = col + 1;
    end
    row = row + 1;
end

contourf(Ds*1000,Ratios,CV')
title('CV contors','interpreter','latex')
xlabel('D(ms)','interpreter','latex')
ylabel('Ratio = N/M','interpreter','latex')

figure()
plot(Ratios,CV(50,:))
title('CV in terms of N/M ratio','interpreter','latex')
xlabel('Ratio = N/M','interpreter','latex')
X = Ratios(round(end/2));
Y = round(0.5*max(CV(50,:)));
text(X,Y,['D = ',num2str(1000*Ds(50)),'ms'],'interpreter','latex');

figure()
plot(Ds*1000,CV(:,10))
title('CV in terms of window width','interpreter','latex')
xlabel('D(ms)','interpreter','latex')
X = 1000*Ds(round(end/2));
Y = round(0.5*max(CV(:,10)));
text(X,Y,['N/M = ',num2str(Ratios(10))],'interpreter','latex');

%% Part2-F
clear
close all
clc

lambda1 = 100;
lambda2 = 45;
tSim = 0.5;
refPeriod = 0.001;
dt = 1/1000;
tVec = 0:dt:tSim-dt;
spikeTimes1 = refPoissonSpikeGen(lambda1, tSim, refPeriod);
spikeTimes1= dt*round(spikeTimes1/dt);
spikeTrain1 = zeros(size(tVec));
for i = 1 : length(spikeTimes1)
    idx = find(tVec == spikeTimes1(i));
    if (length(idx)~=0)
        spikeTrain1(idx) = 1;
    end
end
    
spikeTimes2 = refPoissonSpikeGen(lambda2, tSim, refPeriod);
spikeTimes2= dt*round(spikeTimes2/dt);
spikeTrain2 = zeros(size(tVec));
for i = 1 : length(spikeTimes2)
    idx = find(tVec == spikeTimes2(i));
    if (length(idx)~=0)
        spikeTrain2(idx) = 1;
    end
end
spikeTrain2 = -1 * spikeTrain2;
spikeTrain3 = spikeTrain1 + spikeTrain2;

Ds = (dt:dt:tSim-dt);
Nnets = 1:100;
row = 1;
for D = Ds
    col = 1;
    for Nnet = Nnets
        spikeTrain2 = EI_CoincidenceDitector(spikeTrain1,Nnet,D);
        spike_intervals =  diff(tVec(spikeTrain2==1)) ;
        if (length(spike_intervals) ~= 0)
            CV(row,col) = std(spike_intervals)/(mean(spike_intervals));
        else
            CV(row,col) = 0;
        end
        col = col + 1;
    end
    row = row + 1;
end

contourf(Ds*1000,Nnets,CV')
title('CV contors','interpreter','latex')
xlabel('D(ms)','interpreter','latex')
ylabel('Nnet','interpreter','latex')
figure()
plot(Nnets,CV(50,:))
title('CV in terms of Nnet','interpreter','latex')
xlabel('Nnet = Ne-Ni','interpreter','latex')
X = Nnets(50);
Y = round(0.5*max(CV(50,:)));
text(X,Y,['D = ',num2str(1000*Ds(50)),'ms'],'interpreter','latex');

figure()
plot(Ds*1000,CV(:,10))
title('CV in terms of window width','interpreter','latex')
xlabel('D(ms)','interpreter','latex')
X = 1000*Ds(250);
Y = round(0.5*max(CV(:,10)));
text(X,Y,['Nnet = ',num2str(Nnets(10))],'interpreter','latex');
