%% Part1 - Q1
clear all
close all
clc

Weights = [];
weights = [];
w = 0;
epsilon = 0.1;
trialNumber = 500;

% Pavlovian
for i = 1 : trialNumber
    u = randi(2) - 1;
    r = u;
    v = w*u;
    delta = r - v;
    weights = [weights,w];
    w = w + epsilon*delta*u;
end
Weights = [Weights;weights];

% Extinction
weights = [];
for i = 1 : trialNumber
    u = randi(2) - 1;
    r = 0;
    v = w*u;
    delta = r - v;
    weights = [weights,w];
    w = w + epsilon*delta*u;
end
Weights = [Weights;weights];

% Partial
weights = [];
w = 0;
for i = 1 : trialNumber
    u = randi(2) - 1;
    r = rand(1)>0.5;
    v = w*u;
    delta = r - v;  
    weights = [weights,w];
    w = w + epsilon*delta*u;
end
Weights = [Weights;weights];

% Blocking
weights = [];
W = [0,0];
for i = 1 : trialNumber/2
    u = [randi(2, 1)-1,0];
    r = u(1);
    v = W*u';
    delta = r - v;
    W = W + epsilon*delta.*u;
    weights = [weights,W'];
end
for i = trialNumber/2+1 : trialNumber
    u = randi(2,1,2)-1;
    r = u(1);
    v = W*u';
    delta = r - v;
    W = W + epsilon*delta.*u;
    weights = [weights,W'];
end
Weights = [Weights;weights];

%Inhibitory
weights = [];
W = [0,0];
for i = 1 : trialNumber
    u = randi(2,1,2) - 1;
    r = u(1)==1 &  u(2)==0;
    v = W*u';
    delta = r - v;
    W = W + epsilon*delta.*u;
    weights = [weights,W'];
end
Weights = [Weights;weights];


%Overshadow
weights = [];
W = [0,0];
for i = 1 : trialNumber
    u = randi(2,1,2) - 1;
    r = u(1)==1 &  u(2)==1;
    v = W*u';
    delta = r - v;
    W = W + epsilon*delta.*u;
    weights = [weights,W'];
end
Weights = [Weights;weights];

figure()
set(gcf,'color','w');
subplot(3,1,1)
plot(Weights(1,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Pavlovian's weight over time with u $$\in$$ [0,1] and r = u ",'interpreter','latex');
subplot(3,1,2)
plot(Weights(2,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Extinction's weight over time with u $$\in$$ [0,1] and r = u ",'interpreter','latex');
subplot(3,1,3)
plot(Weights(3,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Patial's weight over time with u $$\in$$ [0,1] and r $$\in$$ [0,1] ",'interpreter','latex');


figure()
set(gcf,'color','w');
subplot(3,1,1)
plot(Weights(4,:));
hold on 
plot(Weights(5,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Blocking's weight over time with u $$\in$$ [0,1] and r = u1 $$\&\&$$ u2 ",'interpreter','latex');
legend('w1','w2','interpreter','latex')

subplot(3,1,2)
plot(Weights(6,:));
hold on 
plot(Weights(7,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Inhibitory's weight over time with u $$\in$$ [0,1] and r = u1 \&\& !u2 ",'interpreter','latex');
legend('w1','w2','interpreter','latex')

subplot(3,1,3)
plot(Weights(8,:));
hold on 
plot(Weights(9,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Overshadow's weight over time with u $$\in$$ [0,1] and r $$\in$$ u1 \&\& u2 ",'interpreter','latex');
legend('w1','w2','interpreter','latex')


%% Part1 - Q2
clear all
close all
clc


Weights = [];
weights = [];
w = 0;
epsilon = [0.01,0.3];
trialNumber = 500;

%Overshadow
weights = [];
W = [0,0];
for i = 1 : trialNumber
    u = randi(2,1,2) - 1;
    r = u(1)==1 &  u(2)==1;
    v = W*u';
    delta = r - v;
    W = W + epsilon.*delta.*u;
    weights = [weights,W'];
end
Weights = [Weights;weights];


weights = [];
W = rand(1,2);
for i = 1 : trialNumber
    u = randi(2,1,2) - 1;
    r = u(1)==1 &  u(2)==1;
    v = W*u';
    delta = r - v;
    W = W + epsilon.*delta.*u;
    weights = [weights,W'];
end
Weights = [Weights;weights];

figure()
set(gcf,'color','w');
plot(Weights(1,:));
hold on 
plot(Weights(2,:));
hold on 
plot(Weights(3,:));
hold on 
plot(Weights(4,:));
ylabel('w(t)','interpreter','latex');
xlabel('trial number','interpreter','latex');
title("Overshadow's weight over time with u $$\epsilon_1 = 0.01, \epsilon_2 = 0.3$$ ",'interpreter','latex');
legend('w1---initialized by zero','w2---initialized by zero','w1---initialized randomly','w2---initialized randomly','interpreter','latex')


%% Part2 - Q1
clear all
close all
clc


tauw = 0.07;
tauv = 0.7;
paradigm = 'backward blocking';
preTrainingTrialNumber = 10;
trainingTrialNumber = 11;
sigma = eye(2)*0.6;
plottingFlag = 'True';
[weights, sigmas, sigmaMat] = KalmanFilter(tauw, tauv, paradigm,preTrainingTrialNumber, trainingTrialNumber, sigma, plottingFlag);

%%
figure()
w1 = -1:0.02:2;
w2 = -1:0.02:2;
[w1,w2] = meshgrid(w1,w2);
X = [reshape(w1,size(w1,1)*size(w1,2),1),reshape(w2,size(w2,1)*size(w2,2),1)];
stages = [1,10,20];
i = 0;
for stage = stages
    i = i + 1;
    y = mvnpdf(X,weights(stage,:),[sigmaMat(1,stage) sigmaMat(3,stage); sigmaMat(4,stage) sigmaMat(2,stage)]);
    y = reshape(y,length(w2),length(w1));
    set(gcf,'Color',[1 1 1]);
    subplot(1,3,i);
    surf(w1,w2,y)
    caxis([min(y,[],'all'),max(y,[],'all')])
    axis([-3 3 -3 3 0 0.4])
    xlim([-1 2])
    ylim([-1 2])
    zlim([-2 10])
    xlabel('w1','Interpreter','latex')
    ylabel('w2','Interpreter','latex')
    zlabel('Joint distribution','Interpreter','latex')
    view([0 90]);
    shading interp
    colormap("gray")
    title("t = "+num2str(stage),'Interpreter','latex');
    axis square
    hold on
    scatter3(weights(stage,1),weights(stage,2),6,'*','black')
end

%% Part2 - Q2
clear all
close all
clc

tauwMat = [0.07,0.7,1.7];
tauvMat = [0.07,0.7,1.7];
paradigm = 'backward blocking';
preTrainingTrialNumber = 10;
trainingTrialNumber = 11;
plottingFlag = 'True';
cnt = 0;
for tauw = tauwMat
    for tauv = tauvMat
        sigma = eye(2)*0.6;
        cnt = cnt + 1;
        weights = [];
        sigmas = [];
        sigmaMat = [];
        w = [0, 0];
        A = eye(2);
        for i = 1 : preTrainingTrialNumber
            weights = [weights; w];
            sigmas = [sigmas; [sigma(1, 1), sigma(2, 2)]];
            sigmaMat = [sigmaMat, [sigma(1,1);sigma(2,2);sigma(1,2);sigma(2,1)]];
            if isequal(paradigm, 'backward blocking')
                C = [1;1];
                r = 1;
                Wnoise = tauw^2;
                Vnoise = tauv^2;
            else
                C = [1;0];
                r = 1;
                Wnoise = [tauw^2,0];
                Vnoise = tauv^2;
            end
            y = r;
            w = w*A;
            sigma = A*sigma*A' + Wnoise;
            G = sigma*C*((C'*sigma*C + Vnoise)^-1);
            sigma = sigma - G*C'*sigma;
            w = w + G'*(y - w*C);
        end
    
        for i = 1 : trainingTrialNumber 
            weights = [weights; w];
            sigmas = [sigmas; [sigma(1, 1), sigma(2, 2)]];
            sigmaMat = [sigmaMat, [sigma(1,1);sigma(2,2);sigma(1,2);sigma(2,1)]];
            Wnoise = tauw^2;
            Vnoise = tauv^2;
            if isequal(paradigm, 'blocking')
                C = [1;1];
                r = 1;
            elseif isequal(paradigm, 'unblocking')
                C = [1;1];
                r = 2;
            elseif isequal(paradigm, 'backward blocking')
                C = [1;0];
                r = 1;
            end
            y = r;
            w = w*A;
            sigma = A*sigma*A' + Wnoise;
            G = sigma*C*((C'*sigma*C + Vnoise)^-1);
    
            sigma = sigma - G*C'*sigma;
            w = w + G'*(y - w*C); 
    
        end
        
    
        trials = 0 : trainingTrialNumber + preTrainingTrialNumber - 1;
        trainingTrials = preTrainingTrialNumber : trainingTrialNumber + preTrainingTrialNumber-1;
        if isequal(plottingFlag, 'True')
            set(gcf,'color','w');
            subplot(2*length(tauvMat),length(tauwMat),cnt);
            plot(trials,weights(:, 1));
            hold on
            plot(trials,weights(:, 2));
            hold on
            Min = min(weights,[],'all');
            Max = 1.1*max(weights,[],'all');
            ylim([Min,Max]);
            plot(ones(1,size(weights,1))*10, linspace(Min,Max,size(weights,1)), 'x','Color','black');
            xlabel('trial number','interpreter', 'latex');
            ylabel('w(t)','interpreter', 'latex');
            legend('$$w_1$$', '$$w_2$$','interpreter', 'latex');
            title(num2str(paradigm) + " weights, process noise = "+num2str(tauw)+", measurement noise = "+num2str(tauv),'interpreter', 'latex');
    
            subplot(2*length(tauvMat),length(tauwMat),cnt+length(tauwMat));
            plot(trials,sigmas(:, 1));
            hold on
            if isequal(paradigm, 'backward blocking')
                plot(trials,sigmas(:, 2));            
            else
                plot(trainingTrials,sigmas(trainingTrials+1, 2));
            end
            hold on
            Min = min(sigmas,[],'all');
            Max = max(sigmas,[],'all');
            ylim([Min,Max]);
            plot(ones(1,size(sigmas,1))*10, linspace(Min,Max,size(sigmas,1)), 'x','Color','black');
            xlabel('trial number','interpreter', 'latex');
            ylabel('$$\sigma^2$$(t)','interpreter', 'latex');
            legend('$$w_1$$', '$$w_2$$','interpreter', 'latex');
            title(num2str(paradigm) + " sigmas, process noise = "+num2str(tauw)+", measurement noise = "+num2str(tauv),'interpreter', 'latex');
            if (rem(cnt,length(tauwMat))==0)
                cnt = cnt + length(tauwMat);
            end
        end
    end
end
%%

paradigm = 'backward blocking';
preTrainingTrialNumber = 10;
trainingTrialNumber = 11;
sigma = eye(2)*0.6;
plottingFlag = 'False';
cnt = 0;
for tauw = tauwMat
    for tauv = tauvMat
        [weights, sigmas, sigmaMat] = KalmanFilter(tauw, tauv, paradigm,preTrainingTrialNumber, trainingTrialNumber, sigma, plottingFlag);
        w1 = -1:0.02:2;
        w2 = -1:0.02:2;
        [w1,w2] = meshgrid(w1,w2);
        X = [reshape(w1,size(w1,1)*size(w1,2),1),reshape(w2,size(w2,1)*size(w2,2),1)];
        stages = [1,10,20];
        i = cnt;
        for stage = stages
            i = i + 1;
            y = mvnpdf(X,weights(stage,:),[sigmaMat(1,stage) sigmaMat(3,stage); sigmaMat(4,stage) sigmaMat(2,stage)]);
            y = reshape(y,length(w2),length(w1));
            set(gcf,'Color',[1 1 1]);
            subplot(length(tauvMat),3*length(tauwMat),i);
            surf(w1,w2,y)
            caxis([min(y,[],'all'),max(y,[],'all')])
            axis([-3 3 -3 3 0 0.4])
            xlim([-1 2])
            ylim([-1 2])
            zlim([-2 10])
            xlabel('w1','Interpreter','latex')
            ylabel('w2','Interpreter','latex')
            zlabel('Joint distribution','Interpreter','latex')
            view([0 90]);
            shading interp
            colormap("gray")
            title("t="+num2str(stage)+",$$\tau_v$$="+num2str(tauv)+",$$\tau_w$$="+num2str(tauw),'Interpreter','latex');
            axis square
            hold on
            scatter3(weights(stage,1),weights(stage,2),6,'*','black')
        end 
        cnt = cnt + 3;
    end
end
%% Part2 - Q3

paradigm = 'blocking';
preTrainingTrialNumber = 10;
trainingTrialNumber = 101;
sigma = eye(2)*0.6;
plottingFlag = 'False';
randomTrial = randi(20);
tauvs = linspace(0, 5, 100);
tauw = 0.07;
Weights1 = [];
Sigmas1 = [];
for tauv = tauvs
    [weights, sigmas,sigmaMat] = KalmanFilter(tauw, tauv, paradigm, preTrainingTrialNumber, trainingTrialNumber, sigma, plottingFlag);
    Weights1 = cat(3, Weights1, weights);
    Sigmas1 = cat(3, Sigmas1, sigmas);
end
tauws = linspace(0, 5, 100);
tauv = 0.7;
Weights2 = [];
Sigmas2 = [];
for tauw = tauws
    [weights, sigmas,sigmaMat] = KalmanFilter(tauw, tauv, paradigm, preTrainingTrialNumber, trainingTrialNumber, sigma, plottingFlag);
    Weights2 = cat(3, Weights2, weights);
    Sigmas2 = cat(3, Sigmas2, sigmas);
end
set(gcf,'color','w');
subplot(2, 1, 1);
plot(tauws, squeeze(Weights1(81+randomTrial, 1, :)));
hold on
plot(tauvs, squeeze(Weights2(81+randomTrial, 1, :)));
title('$$w_1$$ on a random trial over different $$\tau_w$$ and $$\tau_v$$','Interpreter','latex');
xlabel('$\tau$', 'interpreter', 'latex');
ylabel('gain','Interpreter','latex');
legend('gain thorugh $$\tau_w$$','gain thorugh $$\tau_v$$', 'interpreter', 'latex')

subplot(2, 1, 2);
plot(tauws, squeeze(Weights1(80, 2, :)));
hold on
plot(tauvs, squeeze(Weights2(80, 2, :)));
title('$$w_2$$ on on a random trial over different $$\tau_w$$ and $$\tau_v$$','Interpreter','latex');
xlabel('$\tau$', 'interpreter', 'latex');
ylabel('gain','Interpreter','latex');
legend('gain thorugh $$\tau_w$$','gain thorugh $$\tau_v$$', 'interpreter', 'latex')

%% Part2 - Q5
tauw = 0.07;
tauv = 0.7;
paradigm = 'paradigm5';
preTrainingTrialNumber = 10;
trainingTrialNumber = 11;
sigma = eye(1)*0.6;
plottingFlag = 'True';
%
% [weights, sigmas, sigmaMat] = KalmanFilter(tauw, tauv, paradigm,preTrainingTrialNumber, trainingTrialNumber, sigma, plottingFlag);

weights = [];
sigmas = [];
sigmaMat = [];
w = 0;
A = eye(1);
for i = 1 : preTrainingTrialNumber
    weights = [weights; w];
    sigmas = [sigmas;sigma(1, 1)];
    Wnoise = tauw^2;
    Vnoise = tauv^2;
    C = 1;
    r = 1;
    y = r;
    w = w*A;
    sigma = A*sigma*A' + Wnoise;
    G = sigma*C*((C'*sigma*C + Vnoise)^-1);
    sigma = sigma - G*C'*sigma;
    w = w + G'*(y - w*C);
end

for i = 1 : trainingTrialNumber 
    weights = [weights; w];
    sigmas = [sigmas;sigma(1, 1)];
    Wnoise = tauw^2;
    Vnoise = tauv^2;
    C = 1;
    r = -1;
    y = r ;
    w = w*A;
    sigma = A*sigma*A' + Wnoise;
    G = sigma*C*((C'*sigma*C + Vnoise)^-1);

    sigma = sigma - G*C'*sigma;
    w = w + G'*(y - w*C); 

end


trials = 0 : trainingTrialNumber + preTrainingTrialNumber - 1;
trainingTrials = preTrainingTrialNumber : trainingTrialNumber + preTrainingTrialNumber-1;
if isequal(plottingFlag, 'True')
    set(gcf,'color','w');
    subplot(2, 1, 1);
    plot(trials,weights(:, 1));
    hold on
    Min = min(weights,[],'all');
    Max = 1.1*max(weights,[],'all');
    ylim([Min,Max]);
    plot(ones(1,size(weights,1))*10, linspace(Min,Max,size(weights,1)), 'x','Color','black');
    xlabel('trial number','interpreter', 'latex');
    ylabel('w(t)','interpreter', 'latex');
    legend('$$w_1$$', '$$w_2$$','interpreter', 'latex');
    title(num2str(paradigm) + " weights",'interpreter', 'latex');

    subplot(2, 1, 2);
    plot(trials,sigmas(:, 1));
    hold on
    Min = min(sigmas,[],'all');
    Max = max(sigmas,[],'all');
    ylim([Min,Max]);
    plot(ones(1,size(sigmas,1))*10, linspace(Min,Max,size(sigmas,1)), 'x','Color','black');
    xlabel('trial number','interpreter', 'latex');
    ylabel('$$\sigma^2$$(t)','interpreter', 'latex');
    legend('$$w_1$$', '$$w_2$$','interpreter', 'latex');
    title(num2str(paradigm) + " sigmas",'interpreter', 'latex');
end

%% Part3 - Q1
clear all
close all
clc
tauw = 0.07;
tauv = 0.7;
paradigm = 'paradigm5';
trainingTrialNumber = 101;
sigma0 = eye(1)*0.6;
sigma = sigma0;
plottingFlag = 'True';
weights = [];
sigmas = [];
sigmaMat = [];
wreals = [];
rs = [];
w = 0;
r = 0;
wreal = 0;
MSE = 0;
beta = 0;
betas = [];
gamma = 3.3;
A = eye(1);
for i = 1 : trainingTrialNumber
    weights = [weights; w];
    sigmas = [sigmas;sigma(1, 1)];
    wreals = [wreals,wreal];
    betas = [betas,beta];
    rs = [rs,r];
    vnoise = normrnd(0,tauv);
    wnoise = normrnd(0,tauw);
    Wnoise = tauw^2;
    Vnoise = tauv^2;
    binaryVariable = rand(1)<0.05;
    C = 1;
    wreal = wreal*A + wnoise + binaryVariable*normrnd(0,20*tauw);
    r = wreal*C + vnoise;
    w = w*A;
    sigma = A*sigma*A' + Wnoise;
    G = sigma*C*((C'*sigma*C + Vnoise)^-1);
    sigma = sigma - G*C'*sigma;
    w = w + G'*(r - w*C);
    beta = ((r - w*C)^2) / (C'*sigma*C + Vnoise);
    if beta > gamma
        sigma = sigma0;
        sigmas(i) = 10*gamma;
    end
end


trials = 0 : trainingTrialNumber-1;
if isequal(plottingFlag, 'True')
    set(gcf,'color','w');
    subplot(3, 1, 1);
    scatter(trials,wreals,'black','.');
    hold on
    scatter(trials,rs,'black','x');
    xlabel('t','interpreter', 'latex');
    legend('w(t)','r(t)','interpreter','latex','Location','northwest')
    subplot(3, 1, 2);
    scatter(trials,weights,'black','o');
    hold on
    scatter(trials,rs,'black','x');
    xlabel('t','interpreter', 'latex');
    legend('$$\hat w(t)$$','interpreter','latex','Location','northwest')
    subplot(3, 1, 3);
    plot(trials,betas,'k')
    hold on
    plot(trials,sigmas,'--k')
    hold on
    plot(trials,ones(1,trainingTrialNumber)*gamma,'-.k')
    ylim([0 10])
    xlabel('t','interpreter', 'latex');
    legend('ACh','NE','$$\gamma$$','interpreter','latex','Location','northwest')
end
%% MSE calculating
clear all
close all
clc
tauw = 0.07;
tauv = 0.7;
trainingTrialNumber = 100;
gammas = 0:0.5:30;
SSEmat =[];
for iteration = 1 : 100
   iteration
   r = 0;
   rs = [];
   wreals = [];
   wreal = 0;
   A = eye(1);
   vnoiseVector = normrnd(0,tauv,1,trainingTrialNumber);
   wnoiseVector = normrnd(0,tauw,1,trainingTrialNumber);
   for i = 1 : trainingTrialNumber
        wreals = [wreals,wreal];
        rs = [rs,r];
        vnoise = vnoiseVector(i);
        wnoise = wnoiseVector(i);
        Wnoise = tauw^2;
        Vnoise = tauv^2;
        binaryVariable = rand(1)<0.05;

        wreal = wreal*A + wnoise + binaryVariable*normrnd(0,20*tauw);
        C = 1;
        r = wreal*C + vnoise;
    end
    SSEs = [];
    for gamma = gammas
        sigma0 = eye(1)*0.6;
        sigma = sigma0;   
        weights = [];
        sigmas = [];
        sigmaMat = [];
        w = 0;
        beta = 0;
        betas = [];
        for i = 1 : trainingTrialNumber
            weights = [weights; w];
            sigmas = [sigmas;sigma(1, 1)];
            w = w*A;
            sigma = A*sigma*A' + Wnoise;
            G = sigma*C*((C'*sigma*C + Vnoise)^-1);
            sigma = sigma - G*C'*sigma;
            w = w + G'*(rs(i) - w*C);
            beta = ((rs(i) - w*C)^2) / (C'*sigma*C + Vnoise);
            if beta > gamma
                sigma = 100;
            end
        end
        SSE =  sum((wreals - weights').^2);
        SSEs = [SSEs,SSE];    
    end
    SSEmat(iteration,:) = SSEs;
end
SSE = mean(SSEmat);
std = sqrt(var(SSEmat));
set(gcf,'color','w');
errorbar(gammas,SSE,std/40,'k');
%plot(gammas,SSE,'k')
xlabel('$$\gamma$$','interpreter', 'latex');
ylabel('MSE','interpreter', 'latex');

