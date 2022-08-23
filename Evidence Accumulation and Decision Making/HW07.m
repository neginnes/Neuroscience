%% Part1-Q1
clear all
close all
clc

B = 1;
sigma = 1;
dt = 0.1;
timeInterval = 10;
X = 1;
Y = simple_model(B,sigma,dt,timeInterval,X);
decision = strings(1,2);
decision(1) = "No Go";
decision(2) = "Go";
fprintf("The decision is '" + decision(2*Y-1) + "'\n");

%% Part1-Q2
clear all
close all
clc

B = 1;
sigma = 1;
dt = 0.1;
timeInterval = 1;
Xinit = 0;
X1 = [];
Y1 = [];
X2 = [];
Y2 = [];
N1 = 20;
for i = 1:N1
    [Y, X] = simple_model(B,sigma,dt,timeInterval,Xinit);
    X1 = [X1;X(end)];
    Y1 = [Y1;Y];
end
N2 = 1000;
for i = 1:N2
    [Y, X] = simple_model(B,sigma,dt,timeInterval,Xinit);
    X2 = [X2;X(end)];
    Y2 = [Y2;Y];
end
set(gcf,'color','w');
subplot(2, 2, 1);
histogram(Y,'Normalization','probability');
title(['distribution of decision with N = ',num2str(N1)],'Interpreter','latex');
xticks([0 1])
xticklabels({'No Go' 'Go'})

subplot(2, 2, 2);
histogram(Y2,'Normalization','probability');
title(['distribution of decision with N = ',num2str(N2)],'Interpreter','latex');
xticks([0 1])
xticklabels({'No Go' 'Go'})

subplot(2, 2, 3);
histogram(X,'Normalization','probability');
title(['distribution of the decision variable with N = ',num2str(N1),' mean = ', num2str(mean(X1)), ' var = ', num2str(var(X1))],'Interpreter','latex');

subplot(2, 2, 4);
histogram(X2,'Normalization','probability');
title(['distribution of the decision variable with N = ',num2str(N2),' mean = ', num2str(mean(X2)), ' var = ', num2str(var(X2))],'Interpreter','latex');

%% 
clear all 
close all
clc 

B = [-1,0,0.1,1,10];
sigma = 1;
dt = 0.1;
timeInterval = 10;
Xinit = 0;
N = 1000;
for i = 1:length(B)
    Xs = [];
    Ys = [];
    for j = 1:N
        [Y, X] = simple_model(B(i),sigma,dt,timeInterval,Xinit);
        Xs = [Xs;X(end)];
        Ys = [Ys;Y];
    end
    set(gcf,'color','w');
    subplot(5, 2, 2*i-1);
    histogram(Xs,'Normalization','probability');
    title([' distribution of the decision variable with N = ',num2str(N),' and B = ',num2str(B(i)),' mean = ', ...
        num2str(mean(Xs)), ' var = ', num2str(var(Xs))],'Interpreter','latex');
    subplot(5, 2, 2*i);
    histogram(Ys,'Normalization','probability');
    title([' distribution of decision with  N = ',num2str(N),' and B = ',num2str(B(i)),' mean = ', ...
        num2str(mean(Ys)), ' var = ', num2str(var(Ys))],'Interpreter','latex');
    xticks([0 1])
    xticklabels({'No Go' 'Go'})
end
%%

clear all 
close all
clc 

B = [-1,0,0.1,1,10];
sigma = 1;
dt = 0.1;
timeInterval = 10;
Xinit = 0;
for i = 1:length(B)
    Xs = [];
    Ys = [];
    [Y, X] = simple_model(B(i),sigma,dt,timeInterval,Xinit);
    Xs = [Xs;X];
    Ys = [Ys;Y];
    set(gcf,'color','w');
    plot(0:dt:timeInterval,Xs);
    hold on
end
title('time course of the event','Interpreter','latex');
legend('B = -1','B = 0','B = 0.1','B = 1','B = 10','interpreter','latex')
xlabel('time(s)','Interpreter','latex')
ylabel('X(t)','Interpreter','latex')
xlim([0 timeInterval])

%% Part1-Q3

clear all 
close all
clc 

nTrials = 1000;
B = 0.1;
sigma = 1;
dt = 0.1;
timeInterval = linspace(0.5,10,100);

Xinit = 0;
error = [];
for TI = timeInterval
    Ys = [];    
    for trialCount = 1:nTrials
        [Y,~] = simple_model(B,sigma,dt,TI,Xinit);
        Ys = [Ys;Y];
    end
    error = [error,sum(Ys==0)/nTrials];
end

figure()
set(gcf,'color','w');
plot(timeInterval,error*100);
title('error rate through time','Interpreter','latex');
xlabel('time intervals(s)','Interpreter','latex')
ylabel('percentage of error','Interpreter','latex')
xlim([0.5 10])
%% Part1-Q4

clear all 
close all
clc 

nTrials = 20;
B = 0.1;
sigma = 1;
dt = 0.1;
timeInterval = 10;
TIs = 0:dt:timeInterval;
Xinit = 0;
Xs = [];
for trialCount = 1:nTrials
    [Y, X] = simple_model(B,sigma,dt,timeInterval,Xinit);
    Xs = [Xs,X];
    set(gcf,'color','w');
    p1 = plot(TIs,X,'--','LineWidth',0.25);
    p1.HandleVisibility = 'off';
    hold on
end

plot(TIs,mean(Xs,2),'LineWidth',1,'Color','black');
hold on
plot(TIs,mean(Xs,2) + std(Xs,0,2),'LineWidth',1,'Color','blue');
hold on
plot(TIs,mean(Xs,2) - std(Xs,0,2),'LineWidth',1,'Color','red');
hold on

meansTheory = B*TIs;
plot(TIs,meansTheory,'-*','LineWidth',0.5,'Color','black')
hold on
plot(TIs,meansTheory + sqrt((1:length(TIs))*sigma^2*dt^2),'-*','LineWidth',0.5,'Color','blue')
hold on
plot(TIs,meansTheory - sqrt((1:length(TIs))*sigma^2*dt^2),'-*','LineWidth',0.5,'Color','red')

title('trajectories over time','Interpreter','latex');
ylabel('X(t)','Interpreter','latex')
xlabel('time(s)','Interpreter','latex')
xlim([0 timeInterval])
legend('Avg','Avg+$$\sigma$$','Avg-$$\sigma$$','theory','theory','theory','interpreter','latex')

%% Part1-Q5

clear all
close all
clc

B = 0.1;
sigma = 1;
dt = 0.1;
timeInterval = 1;
Xinit = 0;
N = 20;
Ys = [];
for i = 1:N
    Y = simple_model2(B,sigma,dt,timeInterval,Xinit);
    Ys = [Ys,Y];
end
set(gcf,'color','w');
histogram(Ys,'Normalization','probability');
title(['distribution of decision with N = ',num2str(N),', mean = ', num2str(mean(Ys)), ', var = ', num2str(var(Ys))],'Interpreter','latex');
xticks([0 1])
xticklabels({'No Go' 'Go'})

%%

clear all
close all
clc

Bs = [-1,0,0.1,1,10];
sigma = 1;
dt = 0.1;
timeInterval = 1;
Xinit = 0;
N = 20;
cnt = 0;
for B = Bs
    cnt = cnt + 1;
    Ys = [];
    for i = 1:N
        Y = simple_model2(B,sigma,dt,timeInterval,Xinit);
        Ys = [Ys,Y];
    end
    subplot(5,1,cnt)
    set(gcf,'color','w');
    histogram(Ys,'Normalization','probability');
    title(['distribution of decision with N = ',num2str(N),' and B = ',num2str(B),', $$X_0 = $$ ',num2str(Xinit),', mean = ', num2str(mean(Ys)), ', var = ', num2str(var(Ys))],'Interpreter','latex');
    xticks([0 1])
    xticklabels({'No Go' 'Go'})
end

%%
clear all
close all
clc

B = 0.1;
sigma = 1;
dt = 0.1;
timeInterval = 1;
Xinits = [-10,-1,0,1,10];
N = 20;
cnt = 0;
for Xinit = Xinits
    cnt = cnt + 1;
    Ys = [];
    for i = 1:N
        Y = simple_model2(B,sigma,dt,timeInterval,Xinit);
        Ys = [Ys,Y];
    end
    subplot(5,1,cnt)
    set(gcf,'color','w');
    histogram(Ys,'Normalization','probability');
    title(['distribution of decision with N = ',num2str(N),' and B = ',num2str(B),', $$X_0 = $$ ',num2str(Xinit),', mean = ', num2str(mean(Ys)), ', var = ', num2str(var(Ys))],'Interpreter','latex');
    xticks([0 1])
    xticklabels({'No Go' 'Go'})
end

%% Part1-Q6
clear all
close all
clc 

nTrials = 10;
B = 0.01;
sigma = 1;
dt = 0.1;
Xinit = 0;
thp = 2;
thn = -2;
timeRange = -1;
for trialCount = 1 : nTrials
    [~,X,T] = two_choice_trial(B,sigma,dt,thn,thp,Xinit);
    timeRange = max(timeRange,T);
    set(gcf,'color','w');
    plot(linspace(0,T,length(X)),X);
    hold('on');
    title('trajectories over time','Interpreter','latex');
end
plot(0:dt:timeRange,ones(1,length(0:dt:timeRange))*thp,'LineWidth',1,'LineStyle','--');
plot(0:dt:timeRange,ones(1,length(0:dt:timeRange))*thn,'LineWidth',1,'LineStyle','--');
xlabel('time(s)','Interpreter','latex')
ylabel('X(t)','Interpreter','latex')
xlim([0,timeRange]);
ylim([-5,5])

%% Part1-Q7

clear all
close all
clc 

nTrials = 10000;
B = 0.01;
sigma = 1;
dt = 0.1;
Xinit = 0;
thp = 2;
thn = -2;
timeRange = -1;
RT1 = [];
RT2 = [];
RT = [];
for trialCount = 1 : nTrials
    [Y,~,T] = two_choice_trial(B,sigma,dt,thn,thp,Xinit);
    if Y == 1
        RT1 = [RT1,T];
    else 
        RT2 = [RT2,T];
    end
    RT = [RT,T];
end

set(gcf,'color','w');
subplot(3,2,[1,2])
plot(RT1,1,'o','Color','red');
hold on
plot(RT2,-1,'o','Color','blue');
title('decisions and reaction times','Interpreter','latex');
ylim([-2 , 2])

subplot(3,2,3)
histogram(RT1,100,'Normalization','probability')
title('right decision reaction time distribution','Interpreter','latex');

subplot(3,2,4)
histogram(RT2,100,'Normalization','probability')
title('wrong decision reaction time distribution','Interpreter','latex');

subplot(3,2,5)
histogram(RT1,100,'Normalization','probability')
hold on
histogram(RT2,100,'Normalization','probability')
legend('right','wrong','Interpreter','latex');
title('wrong decision Vs right decision reaction time distribution','Interpreter','latex');

subplot(3,2,6)
histogram(RT,100,'Normalization','probability')
title('reaction time distribution','Interpreter','latex');

figure()
set(gcf,'color','w');
subplot(2,2,1)
histfit(RT1,100,'inverse gaussian','Normalization','probability')
legend('data distribution','inverse gaussian','interpreter','latex')
title('right decision reaction time distribution','Interpreter','latex');

subplot(2,2,2)
histfit(RT2,100,'inverse gaussian','Normalization','probability')
legend('data distribution','inverse gaussian','interpreter','latex')
title('wrong decision reaction time distribution','Interpreter','latex');

subplot(2,2,3)
histogram(RT1,100,'Normalization','probability')
hold on
histfit(RT2,100,'inverse gaussian','Normalization','probability')
legend('right','wrong','inverse gaussian','interpreter','latex');
title('wrong decision Vs right decision reaction time distribution','Interpreter','latex');

subplot(2,2,4)
histfit(RT,100,'inverse gaussian','Normalization','probability')
legend('data distribution','inverse gaussian','interpreter','latex')
title('reaction time distribution','Interpreter','latex');

%% Part1-Q8
clear all
close all
clc 

nTrials = 2;
B1 = -0.01;
B2 = 0.01;
sigma1 = 1;
sigma2 = 1;
dt = 0.1;
Xinit1 = 0;
Xinit2 = 0;
th1n = -2;
th2p = 2;
timeRange = -1;
c = strings(2,1);
c(1) = 'blue';
c(2) = 'red';
for trialCount = 1 : nTrials
    [Y,X,X1,X2,T] = race_trial(B1,B2,sigma1,sigma2,dt,th1n,th2p,Xinit1,Xinit2);
    timeRange = max(timeRange,T);
    set(gcf,'color','w');
    plot(linspace(0,T,length(X1)),X1,'Color',c(trialCount));
    hold on
    plot(linspace(0,T,length(X2)),X2,'LineStyle','--','Color',c(trialCount));
    hold('on');
    plot(T,X(end),'Color',c(trialCount),'Marker','*','LineWidth',2);
    title('trajectories over time','Interpreter','latex');
end
plot(0:dt:timeRange,ones(1,length(0:dt:timeRange))*th2p,'LineWidth',1,'LineStyle','--','Color','black');
plot(0:dt:timeRange,ones(1,length(0:dt:timeRange))*th1n,'LineWidth',1,'LineStyle','--','Color','black');
legend('X1-trial1','X2-trial1','Y-trial1','X1-trial2','X2-trial2','Y-trial2','$$\theta^+$$','$$\theta^-$$','interpreter','latex')
xlabel('time(s)','Interpreter','latex')
xlim([0,timeRange]);
ylim([-10 10])

%% Part1-Q9
clear all
close all
clc 

nTrials = 3;
B1 = -0.05;
B2 = 0.05;
sigma1 = 1;
sigma2 = 1;
dt = 0.1;
Xinit1 = 0;
Xinit2 = 0;
th1n = -2;
th2p = 2;
timeInterval = 10;
timeRange = -1;
c = strings(3,1);
c(1) = 'blue';
c(2) = 'red';
c(3) = 'green';
for trialCount = 1 : nTrials
    [Y,X,X1,X2,T] = race_trial2(B1,B2,sigma1,sigma2,dt,th1n,th2p,Xinit1,Xinit2,timeInterval);
    timeRange = max(timeRange,T);
    set(gcf,'color','w');
    plot(linspace(0,T,length(X1)),X1,'Color',c(trialCount));
    hold on
    plot(linspace(0,T,length(X2)),X2,'LineStyle','--','Color',c(trialCount));
    hold('on');
    th = [th1n,th2p];
    plot(T,th(Y+1),'Color',c(trialCount),'Marker','*','LineWidth',2);
    title('trajectories over time','Interpreter','latex');
end
plot(0:dt:timeRange,ones(1,length(0:dt:timeRange))*th2p,'LineWidth',1,'LineStyle','--','Color','black');
plot(0:dt:timeRange,ones(1,length(0:dt:timeRange))*th1n,'LineWidth',1,'LineStyle','--','Color','black');
legend('X1-trial1','X2-trial1','Y-trial1','X1-trial2','X2-trial2','Y-trial2','X1-trial3','X2-trial3,Y-trial3','$$\theta^+$$','$$\theta^+$$','interpreter','latex')
xlabel('time(s)','Interpreter','latex')
xlim([0,timeRange]);
ylim([-10 10])

%% Part2-Q1
clear all
close all
clc

% sample 1
LIP_threshold = 1000;
M = 5;
MT_p_values = [0.1 0.1];
LIP_weights = [0.5 -0.5];

[MT,LIP,~,rates,tVec] = lip_activity(MT_p_values,LIP_weights,LIP_threshold,M);

set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 2000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 2000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP, [150 2000]), []);
title('LIP','Interpreter','latex');  
subplot(6, 1, [4 5 6]);
plot(tVec,rates/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%% 
clear all 
close all 
clc

% sample 2
LIP_threshold = 1000;
M = 50;
MT_p_values = [0.2 0.01];
LIP_weights = [0.5 -0.5];

[MT,LIP,~,rates,tVec] = lip_activity(MT_p_values,LIP_weights,LIP_threshold,M);

set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 2000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 2000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP, [150 2000]), []);
title('LIP','Interpreter','latex');  
subplot(6, 1, [4 5 6]);
plot(tVec,rates/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%%
clear all 
close all 
clc

% sample 3
LIP_threshold = 1000;
M = 50;
MT_p_values = [0.2 0.2];
LIP_weights = [0.5 -0.05];

[MT,LIP,~,rates,tVec] = lip_activity(MT_p_values,LIP_weights,LIP_threshold,M);

set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 2000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 2000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP, [150 2000]), []);
title('LIP','Interpreter','latex');  
subplot(6, 1, [4 5 6]);
plot(tVec,rates/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])  

%%
clear all 
close all 
clc

% sample 4
LIP_threshold = 1000;
M = 5;
MT_p_values = [0.2 0.2];
LIP_weights = [0.05 -0.5];

[MT,LIP,~,rates,tVec] = lip_activity(MT_p_values,LIP_weights,LIP_threshold,M);

set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 2000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 2000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP, [150 2000]), []);
title('LIP','Interpreter','latex');  
subplot(6, 1, [4 5 6]);
plot(tVec,rates/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%%
clear all 
close all 
clc

% sample 5
LIP_threshold = 1000;
M = 10;
MT_p_values = [0.2 0.2];
LIP_weights = [0.1 -0.1];

[MT,LIP,~,rates,tVec] = lip_activity(MT_p_values,LIP_weights,LIP_threshold,M);

set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 2000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 2000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP, [150 2000]), []);
title('LIP','Interpreter','latex');  
subplot(6, 1, [4 5 6]);
plot(tVec,rates/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%% Part2-Q2
clear all 
close all 
clc

% sample 1
times = 0:0.001:0.5;
LIP_threshold = [1000,1000];
M = 5;
MT_p_values = [ones(size(times))*0.2;ones(size(times))*0.2];
LIP_weights = [0.1 -0.1];

figure()
set(gcf,'color','w');
subplot(2, 1, 1)
plot(times,MT_p_values(1,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');
subplot(2, 1, 2)
plot(times,MT_p_values(2,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');


[MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M);

figure()
set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 5000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 5000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP(:, 1)', [150 5000]), []);
title('LIP1','Interpreter','latex');  
subplot(6, 1, 4);
imshow(imresize(LIP(:, 2)', [150 5000]), []);
title('LIP2','Interpreter','latex'); 
subplot(6, 1, 5);
plot(tVec,rates1/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP1 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

subplot(6, 1, 6);
plot(tVec,rates2/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP2 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%%
clear all 
close all 
clc

% sample 2
times = 0:0.001:0.5;
LIP_threshold = [1000,1000];
M = 5;
MT_p_values = [0.01 + 0.1*(times>0.2); 0.01 + 0.1*(times<0.2)];
LIP_weights = [0.1 -0.1];

figure()
set(gcf,'color','w');
subplot(2, 1, 1)
plot(times,MT_p_values(1,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');
subplot(2, 1, 2)
plot(times,MT_p_values(2,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');


[MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M);

figure()
set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 5000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 5000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP(:, 1)', [150 5000]), []);
title('LIP1','Interpreter','latex');  
subplot(6, 1, 4);
imshow(imresize(LIP(:, 2)', [150 5000]), []);
title('LIP2','Interpreter','latex'); 
subplot(6, 1, 5);
plot(tVec,rates1/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP1 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

subplot(6, 1, 6);
plot(tVec,rates2/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP2 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])


%%
clear all 
close all 
clc

% sample 3
times = 0:0.001:0.5;
LIP_threshold = [10000,10000];
M = 5;
MT_p_values = [0.01 + 0.1*(times<0.2); 0.01 + 0.1*(times>0.2)];
LIP_weights = [0.1 -0.1];

figure()
set(gcf,'color','w');
subplot(2, 1, 1)
plot(times,MT_p_values(1,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');
subplot(2, 1, 2)
plot(times,MT_p_values(2,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');


[MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M);

figure()
set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 5000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 5000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP(:, 1)', [150 5000]), []);
title('LIP1','Interpreter','latex');  
subplot(6, 1, 4);
imshow(imresize(LIP(:, 2)', [150 5000]), []);
title('LIP2','Interpreter','latex'); 
subplot(6, 1, 5);
plot(tVec,rates1/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP1 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

subplot(6, 1, 6);
plot(tVec,rates2/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP2 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%% 
clear all 
close all 
clc

% sample 4
times = 0:0.001:0.5;
LIP_threshold = [1000,1000];
M = 5;
MT_p_values = [ 0.01 + 0.1*sin(pi*times); 0.01 + 0.1*cos(pi*times)];
LIP_weights = [0.1 -0.1];

figure()
set(gcf,'color','w');
subplot(2, 1, 1)
plot(times,MT_p_values(1,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');
subplot(2, 1, 2)
plot(times,MT_p_values(2,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');


[MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M);

figure()
set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 5000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 5000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP(:, 1)', [150 5000]), []);
title('LIP1','Interpreter','latex');  
subplot(6, 1, 4);
imshow(imresize(LIP(:, 2)', [150 5000]), []);
title('LIP2','Interpreter','latex'); 
subplot(6, 1, 5);
plot(tVec,rates1/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP1 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

subplot(6, 1, 6);
plot(tVec,rates2/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP2 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%%
clear all 
close all 
clc

% sample 5
times = 0:0.001:0.5;
LIP_threshold = [1000,1000];
M = 5;
MT_p_values = [ 0.01 + 0.1*cos(pi*times); 0.01 + 0.1*sin(pi*times)];
LIP_weights = [0.1 -0.1];

figure()
set(gcf,'color','w');
subplot(2, 1, 1)
plot(times,MT_p_values(1,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');
subplot(2, 1, 2)
plot(times,MT_p_values(2,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');


[MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M);

figure()
set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 5000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 5000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP(:, 1)', [150 5000]), []);
title('LIP1','Interpreter','latex');  
subplot(6, 1, 4);
imshow(imresize(LIP(:, 2)', [150 5000]), []);
title('LIP2','Interpreter','latex'); 
subplot(6, 1, 5);
plot(tVec,rates1/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP1 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

subplot(6, 1, 6);
plot(tVec,rates2/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP2 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

%%
clear all 
close all 
clc

% sample 6
times = 0:0.001:0.5;
LIP_threshold = [1000,1000];
M = 5;
MT_p_values = [ 0.01 + 0.05*sin(10*pi*times); 0.01 + 0.05*cos(10*pi*times)];
LIP_weights = [0.1 -0.1];

figure()
set(gcf,'color','w');
subplot(2, 1, 1)
plot(times,MT_p_values(1,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');
subplot(2, 1, 2)
plot(times,MT_p_values(2,:));
title('$$MT_1$$(excitatory) Probabily for Response','Interpreter','latex');
xlabel('time(s)','Interpreter','latex');


[MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M);

figure()
set(gcf,'color','w');
subplot(6, 1, 1);
imshow(imresize(MT(:, 1)', [150 5000]), []);
title('$$MT_1$$(excitatory)','Interpreter','latex');
subplot(6, 1, 2);
imshow(imresize(MT(:, 2)', [150 5000]), []);
title('$$MT_2$$(inhibitory)','Interpreter','latex');
subplot(6, 1, 3);
imshow(imresize(LIP(:, 1)', [150 5000]), []);
title('LIP1','Interpreter','latex');  
subplot(6, 1, 4);
imshow(imresize(LIP(:, 2)', [150 5000]), []);
title('LIP2','Interpreter','latex'); 
subplot(6, 1, 5);
plot(tVec,rates1/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP1 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

subplot(6, 1, 6);
plot(tVec,rates2/1000)
xlabel('time(s)','Interpreter','latex')
ylabel('rate(kHz)','Interpreter','latex')
title('LIP2 Firing Rate','Interpreter','latex')
xlim([0,tVec(end)])

