%% Q1,Q2
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
catPos = [8.5,3.5];

% learning
nTrials = 10;
nActs = 4;
eta = 1;
gamma = 1;
uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
uniformProbs(1,:,:) = 1/(nActs-1);
uniformProbs(1,:,3) = 0;
uniformProbs(mapSize,:,:) = 1/(nActs-1);
uniformProbs(mapSize,:,1) = 0;
uniformProbs(:,1,:) = 1/(nActs-1);
uniformProbs(:,1,2) = 0;
uniformProbs(:,mapSize,:) = 1/(nActs-1);
uniformProbs(:,mapSize,4) = 0;
uniformProbs(1,1,:) = 1/(nActs-2);
uniformProbs(1,1,2:3) = 0;
uniformProbs(1,mapSize,:) = 1/(nActs-2);
uniformProbs(1,mapSize,3:4) = 0;
uniformProbs(mapSize,1,:) = 1/(nActs-2);
uniformProbs(mapSize,1,1:2) = 0;
uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
uniformProbs(mapSize,mapSize,1:3:4) = 0;
actProbs = uniformProbs;
mapValues = zeros(mapSize,mapSize);
reward = zeros(mapSize,mapSize);
reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
m = zeros(size(actProbs));

writerObj = VideoWriter("demo1");
writerObj.FrameRate = 20;
open(writerObj);
fig = figure();
for trialCount = 1 : nTrials
    ratPos = randi(mapSize,1,2);
    flag = 0;
    cnt = 0;
    while(1)
        subplot(2,6,[1,2,3,7,8,9])
        mapGenerator(mapSize,targetPos,catPos,ratPos-0.5);
        if(ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2))
            flag = 1;
            text(floor(mapSize/4),floor(3/4*mapSize),'!Reward Catched!','Color','red','Interpreter','latex','FontSize',12);
            log = "Reward";
            pause(0.5)
        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
            flag = 1;
            text(floor(1*mapSize/3),floor(3/4*mapSize),'!Attacked!','Color','red','Interpreter','latex','FontSize',12);
            log = "Attacked";
            pause(0.5)
        end
        title("Main Map at Trial = "+ num2str(trialCount) +", Step = " + num2str(cnt),'Interpreter','latex');
        subplot(2,6,[5,6])
        contourf((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
        title("Value Contours",'Interpreter','latex');
        colorbar()
        colormap("gray")

        subplot(2,6,[11,12])
        [px,py] = gradient(mapValues');  
        contour((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
        hold on
        quiver((1:mapSize)-0.5,(1:mapSize)-0.5,px,py)
        hold off
        title("Gradient",'Interpreter','latex');
        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
        actApproval = 0;
        while(actApproval == 0)
            actApproval = 1;
            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                actApproval = 0;
            end
        end

        if (cnt ~= 0)
            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
            for Action = [1,2,3,4]
                if(Action ~= actPre)
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                else
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                end
            end
            m = edgeHandler(m,mapSize);
            softmax_func = @(x) exp(x)/sum(exp(x));
            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
        end

        if(flag == 1)
            break;
        end
        ratPosPre = ratPos;
        actPre = act;
        ratPos = ratPos + [acti,actj];
        pause(0.2)
        cnt = cnt + 1;
        frame = getframe(fig);
        writeVideo(writerObj,frame);
    end
    "Trial Number: " + num2str(trialCount) + ", Total Steps: " + num2str(cnt) + ", State: " + log  
end
close(writerObj)
%% Q1,Q2 - path plot
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
catPos = [8.5,3.5];

% learning
nTrials = 500;
nActs = 4;
eta = 1;
gamma = 1;
uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
uniformProbs(1,:,:) = 1/(nActs-1);
uniformProbs(1,:,3) = 0;
uniformProbs(mapSize,:,:) = 1/(nActs-1);
uniformProbs(mapSize,:,1) = 0;
uniformProbs(:,1,:) = 1/(nActs-1);
uniformProbs(:,1,2) = 0;
uniformProbs(:,mapSize,:) = 1/(nActs-1);
uniformProbs(:,mapSize,4) = 0;
uniformProbs(1,1,:) = 1/(nActs-2);
uniformProbs(1,1,2:3) = 0;
uniformProbs(1,mapSize,:) = 1/(nActs-2);
uniformProbs(1,mapSize,3:4) = 0;
uniformProbs(mapSize,1,:) = 1/(nActs-2);
uniformProbs(mapSize,1,1:2) = 0;
uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
uniformProbs(mapSize,mapSize,1:3:4) = 0;
actProbs = uniformProbs;
mapValues = zeros(mapSize,mapSize);
reward = zeros(mapSize,mapSize);
reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
m = zeros(size(actProbs));

for trialCount = 1 : nTrials
    ratPos = randi(mapSize,1,2);
    ratPosPre = ratPos;
    flag = 0;
    cnt = 0;
    while(1)
        subplot(2,6,[1,2,3,7,8,9])
        pathGenerator(mapSize,targetPos,catPos,ratPos-0.5,ratPosPre-0.5);
        hold on
        if(ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2))
            flag = 1;
            text(floor(mapSize/4),floor(3/4*mapSize),'!Reward Catched!','Color','red','Interpreter','latex','FontSize',12);
            log = "Reward";
            pause(0.5)
        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
            flag = 1;
            text(floor(1*mapSize/3),floor(3/4*mapSize),'!Attacked!','Color','red','Interpreter','latex','FontSize',12);
            log = "Attacked";
            pause(0.5)
        end
        title("Main Map at Trial = "+ num2str(trialCount+500) +", Step = " + num2str(cnt),'Interpreter','latex');
        subplot(2,6,[5,6])
        contourf((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
        title("Value Contours",'Interpreter','latex');
        colorbar()
        colormap("gray")

        subplot(2,6,[11,12])
        [px,py] = gradient(mapValues');  
        contour((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
        hold on
        quiver((1:mapSize)-0.5,(1:mapSize)-0.5,px,py)
        hold off
        title("Gradient",'Interpreter','latex');
        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
        actApproval = 0;
        while(actApproval == 0)
            actApproval = 1;
            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                actApproval = 0;
            end
        end

        if (cnt ~= 0)
            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
            for Action = [1,2,3,4]
                if(Action ~= actPre)
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                else
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                end
            end
            m = edgeHandler(m,mapSize);
            softmax_func = @(x) exp(x)/sum(exp(x));
            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
        end

        if(flag == 1)
            break;
        end
        ratPosPre = ratPos;
        actPre = act;
        ratPos = ratPos + [acti,actj];
        pause(0.0001)
        cnt = cnt + 1;
    end
    hold off
end


%% Q3
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
catPos = [8.5,3.5];

% learning
nTrials = 500;
nActs = 4;
etas = [0.001,0.01,0.1,1];
gammas = [0.001,0.01,0.1,1];
HitMap = zeros(length(etas),length(gammas));
AverageHitMap = HitMap;
for runs = 1 : 10
    runs
    cnt1 = 1;
    for gamma = gammas
        cnt2 = 1;
        for eta = etas
            uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
            uniformProbs(1,:,:) = 1/(nActs-1);
            uniformProbs(1,:,3) = 0;
            uniformProbs(mapSize,:,:) = 1/(nActs-1);
            uniformProbs(mapSize,:,1) = 0;
            uniformProbs(:,1,:) = 1/(nActs-1);
            uniformProbs(:,1,2) = 0;
            uniformProbs(:,mapSize,:) = 1/(nActs-1);
            uniformProbs(:,mapSize,4) = 0;
            uniformProbs(1,1,:) = 1/(nActs-2);
            uniformProbs(1,1,2:3) = 0;
            uniformProbs(1,mapSize,:) = 1/(nActs-2);
            uniformProbs(1,mapSize,3:4) = 0;
            uniformProbs(mapSize,1,:) = 1/(nActs-2);
            uniformProbs(mapSize,1,1:2) = 0;
            uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
            uniformProbs(mapSize,mapSize,1:3:4) = 0;
            actProbs = uniformProbs;
            mapValues = zeros(mapSize,mapSize);
            reward = zeros(mapSize,mapSize);
            reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
            reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
            m = zeros(size(actProbs));
            steps = [];
            for trialCount = 1 : nTrials
                ratPos = randi(mapSize,1,2);
                flag = 0;
                rflag = 0;
                cnt = 0;
                while(1)
                    if(ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2))
                        flag = 1;
                        rflag = 1;
                    elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
                        flag = 1;
                    end
                    mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
                    actApproval = 0;
                    while(actApproval == 0)
                        actApproval = 1;
                        [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
                        if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                            actApproval = 0;
                        end
                    end
                
                    if (cnt ~= 0)
                        valuePre = mapValues(ratPosPre(1),ratPosPre(2));
                        delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
                        for Action = [1,2,3,4]
                            if(Action ~= actPre)
                                m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                            else
                                m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
            
                            end
                        end
                        m = edgeHandler(m,mapSize);
                        softmax_func = @(x) exp(x)/sum(exp(x));
                        actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
                    end
            
                    if(flag == 1)
                        break;
                    end
                    ratPosPre = ratPos;
                    actPre = act;
                    ratPos = ratPos + [acti,actj];
                    cnt = cnt + 1;
                end
                if (rflag == 1)
                    steps = [steps,cnt-1];
                end
            end
            HitMap(cnt1,cnt2) = mean(steps(end-50:end));
            cnt2 = cnt2 + 1;
        end 
        cnt1 = cnt1 + 1;
    end
    AverageHitMap = AverageHitMap + HitMap;
end
AverageHitMap = AverageHitMap/runs;
set(gcf,'color','w');
imagesc(log10(gammas),log10(etas),AverageHitMap);
title('HeatMap','Interpreter','latex')
ylabel('$$\gamma$$(Logarithmic)','Interpreter','latex')
xlabel('$$\epsilon$$(Logarithmic)','Interpreter','latex')
c = colorbar;
c.Label.String = 'Average Test Steps';
colormap("gray")

%% Q4-1

clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
targetPos2 = [2.5,2.5];
catPos = [8.5,3.5];

% learning
nTrials = 100;
nActs = 4;
eta = 1;
gamma = 1;
uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
uniformProbs(1,:,:) = 1/(nActs-1);
uniformProbs(1,:,3) = 0;
uniformProbs(mapSize,:,:) = 1/(nActs-1);
uniformProbs(mapSize,:,1) = 0;
uniformProbs(:,1,:) = 1/(nActs-1);
uniformProbs(:,1,2) = 0;
uniformProbs(:,mapSize,:) = 1/(nActs-1);
uniformProbs(:,mapSize,4) = 0;
uniformProbs(1,1,:) = 1/(nActs-2);
uniformProbs(1,1,2:3) = 0;
uniformProbs(1,mapSize,:) = 1/(nActs-2);
uniformProbs(1,mapSize,3:4) = 0;
uniformProbs(mapSize,1,:) = 1/(nActs-2);
uniformProbs(mapSize,1,1:2) = 0;
uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
uniformProbs(mapSize,mapSize,1:3:4) = 0;
actProbs = uniformProbs;
mapValues = zeros(mapSize,mapSize);
reward = zeros(mapSize,mapSize);
reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
reward(targetPos2(1)+0.5,targetPos2(2)+0.5) = 1;
m = zeros(size(actProbs));

for trialCount = 1 : nTrials
    ratPos = randi(mapSize,1,2);
    flag = 0;
    cnt = 0;
    while(1)
        if(trialCount == nTrials)
            subplot(2,6,[1,2,3,7,8,9])
            mapGenerator(mapSize,targetPos,catPos,ratPos-0.5);
            hold on
            mapGenerator(mapSize,targetPos2,catPos,ratPos-0.5);
            hold off
        end
        if( (ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2) ) || (ratPos(1)-0.5 == targetPos2(1) && ratPos(2)-0.5 == targetPos2(2)) )
            flag = 1;
            if(trialCount == nTrials)
                str = "!Reward "+ num2str(reward(ratPos(1),ratPos(2))) +" Catched!";
                text(floor(mapSize/4),floor(3/4*mapSize),str,'Color','red','Interpreter','latex','FontSize',12);
                log = "Reward "+ num2str(reward(ratPos(1),ratPos(2)));
            end
        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
            flag = 1;
            if(trialCount == nTrials)
                text(floor(1*mapSize/3),floor(3/4*mapSize),'!Attacked!','Color','red','Interpreter','latex','FontSize',12);
                log = "Attacked";
            end
        end
        if(trialCount == nTrials)
            ylabel("Target1 Value = 1, Target2 Value = 1",'Interpreter','latex');
            title("Main Map After = "+ num2str(trialCount) +", Trials, Steps = " + num2str(cnt) ,'Interpreter','latex');
            subplot(2,6,[5,6])
            contourf((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
            title("Value Contours",'Interpreter','latex');
            colorbar()
            colormap("gray")

            subplot(2,6,[11,12])
            [px,py] = gradient(mapValues');  
            contour((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
            hold on
            quiver((1:mapSize)-0.5,(1:mapSize)-0.5,px,py)
            hold off
            title("Gradient",'Interpreter','latex');
        end
        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
        actApproval = 0;
        while(actApproval == 0)
            actApproval = 1;
            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                actApproval = 0;
            end
        end

        if (cnt ~= 0)
            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
            for Action = [1,2,3,4]
                if(Action ~= actPre)
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                else
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                end
            end
            m = edgeHandler(m,mapSize);
            softmax_func = @(x) exp(x)/sum(exp(x));
            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
        end

        if(flag == 1)
            break;
        end
        ratPosPre = ratPos;
        actPre = act;
        ratPos = ratPos + [acti,actj];
        cnt = cnt + 1;
    end
end

%% Q4-2 
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
targetPos2 = [2.5,2.5];
catPos = [8.5,3.5];

% learning
nTrials = 500;
nActs = 4;
etas = [0.001,0.01,0.1,1];
gammas = [0.001,0.01,0.1,1];
reward = zeros(mapSize,mapSize);
reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
rewardVals = [1,1.5,2,10];
for k = 1 : 4
    reward(targetPos(1)+0.5,targetPos(2)+0.5) = rewardVals(k);
    reward(targetPos2(1)+0.5,targetPos2(2)+0.5) = 1;
    HitMap = zeros(length(etas),length(gammas));
    AverageHitMap = HitMap;
    for runs = 1 : 10
        runs
        cnt1 = 1;
        for gamma = gammas
            cnt2 = 1;
            for eta = etas
                uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
                uniformProbs(1,:,:) = 1/(nActs-1);
                uniformProbs(1,:,3) = 0;
                uniformProbs(mapSize,:,:) = 1/(nActs-1);
                uniformProbs(mapSize,:,1) = 0;
                uniformProbs(:,1,:) = 1/(nActs-1);
                uniformProbs(:,1,2) = 0;
                uniformProbs(:,mapSize,:) = 1/(nActs-1);
                uniformProbs(:,mapSize,4) = 0;
                uniformProbs(1,1,:) = 1/(nActs-2);
                uniformProbs(1,1,2:3) = 0;
                uniformProbs(1,mapSize,:) = 1/(nActs-2);
                uniformProbs(1,mapSize,3:4) = 0;
                uniformProbs(mapSize,1,:) = 1/(nActs-2);
                uniformProbs(mapSize,1,1:2) = 0;
                uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
                uniformProbs(mapSize,mapSize,1:3:4) = 0;
                actProbs = uniformProbs;
                mapValues = zeros(mapSize,mapSize);
                m = zeros(size(actProbs));
                steps = [];
                for trialCount = 1 : nTrials
                    ratPos = randi(mapSize,1,2);
                    flag = 0;
                    rflag = 0;
                    cnt = 0;
                    while(1)
                        if( (ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2) ) || (ratPos(1)-0.5 == targetPos2(1) && ratPos(2)-0.5 == targetPos2(2)) )
                            flag = 1;
                            rflag = 1;
                        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
                            flag = 1;
                        end
                        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
                        actApproval = 0;
                        while(actApproval == 0)
                            actApproval = 1;
                            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
                            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                                actApproval = 0;
                            end
                        end
                    
                        if (cnt ~= 0)
                            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
                            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
                            for Action = [1,2,3,4]
                                if(Action ~= actPre)
                                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                                else
                                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                
                                end
                            end
                            m = edgeHandler(m,mapSize);
                            softmax_func = @(x) exp(x)/sum(exp(x));
                            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
                        end
                
                        if(flag == 1)
                            break;
                        end
                        ratPosPre = ratPos;
                        actPre = act;
                        ratPos = ratPos + [acti,actj];
                        cnt = cnt + 1;
                    end
                    if (rflag == 1)
                        steps = [steps,cnt-1];
                    end
                end
                HitMap(cnt1,cnt2) = mean(steps(end-50:end));
                cnt2 = cnt2 + 1;
            end 
            cnt1 = cnt1 + 1;
        end
        AverageHitMap = AverageHitMap + HitMap;
    end
    AverageHitMap = AverageHitMap/runs;
    subplot(2,2,k)
    set(gcf,'color','w');
    imagesc(log10(gammas),log10(etas),AverageHitMap);
    title("Target1 Value = 1, Target2 Value = "+num2str(rewardVals(k)),'Interpreter','latex')
    ylabel('$$\gamma$$(Logarithmic)','Interpreter','latex')
    xlabel('$$\epsilon$$(Logarithmic)','Interpreter','latex')
    c = colorbar;
    c.Label.String = 'Average Test Steps';
    colormap("gray")
end

%% Q5-1
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
catPos = [8.5,3.5];

% learning
nTrials = 10;
nActs = 4;
eta = 1;
gamma = 1;
lambda = 0.5;
uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
uniformProbs(1,:,:) = 1/(nActs-1);
uniformProbs(1,:,3) = 0;
uniformProbs(mapSize,:,:) = 1/(nActs-1);
uniformProbs(mapSize,:,1) = 0;
uniformProbs(:,1,:) = 1/(nActs-1);
uniformProbs(:,1,2) = 0;
uniformProbs(:,mapSize,:) = 1/(nActs-1);
uniformProbs(:,mapSize,4) = 0;
uniformProbs(1,1,:) = 1/(nActs-2);
uniformProbs(1,1,2:3) = 0;
uniformProbs(1,mapSize,:) = 1/(nActs-2);
uniformProbs(1,mapSize,3:4) = 0;
uniformProbs(mapSize,1,:) = 1/(nActs-2);
uniformProbs(mapSize,1,1:2) = 0;
uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
uniformProbs(mapSize,mapSize,1:3:4) = 0;
actProbs = uniformProbs;
mapValues = zeros(mapSize,mapSize);
reward = zeros(mapSize,mapSize);
reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
m = zeros(size(actProbs));


writerObj = VideoWriter("demo2");
writerObj.FrameRate = 20;
open(writerObj);
fig = figure();
for trialCount = 1 : nTrials
    visitedLocs = [];
    ratPos = randi(mapSize,1,2);
    flag = 0;
    cnt = 0;
    while(1)
        visitedLocs = [visitedLocs;ratPos];
        subplot(2,6,[1,2,3,7,8,9])
        mapGenerator(mapSize,targetPos,catPos,ratPos-0.5);
        if(ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2))
            flag = 1;
            text(floor(mapSize/4),floor(3/4*mapSize),'!Reward Catched!','Color','red','Interpreter','latex','FontSize',12);
            log = "Reward";
            pause(0.5)
        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
            flag = 1;
            text(floor(1*mapSize/3),floor(3/4*mapSize),'!Attacked!','Color','red','Interpreter','latex','FontSize',12);
            log = "Attacked";
            pause(0.5)
        end
        title("Main Map at Trial = "+ num2str(trialCount) +", Step = " + num2str(cnt),'Interpreter','latex');
        subplot(2,6,[5,6])
        contourf((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
        title("Value Contours",'Interpreter','latex');
        colorbar()
        colormap("gray")

        subplot(2,6,[11,12])
        [px,py] = gradient(mapValues');  
        contour((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
        hold on
        quiver((1:mapSize)-0.5,(1:mapSize)-0.5,px,py)
        hold off
        title("Gradient",'Interpreter','latex');
        mapValuesBefore = mapValues;
        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
        mapValues = TDLamdaUPdator(mapValuesBefore,mapValues,visitedLocs,ratPos,lambda);
        actApproval = 0;
        while(actApproval == 0)
            actApproval = 1;
            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                actApproval = 0;
            end
        end

        if (cnt ~= 0)
            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
            for Action = [1,2,3,4]
                if(Action ~= actPre)
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                else
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                end
            end
            m = edgeHandler(m,mapSize);
            softmax_func = @(x) exp(x)/sum(exp(x));
            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
        end

        if(flag == 1)
            break;
        end
        ratPosPre = ratPos;
        actPre = act;
        ratPos = ratPos + [acti,actj];
        pause(0.2)
        cnt = cnt + 1;
        frame = getframe(fig);
        writeVideo(writerObj,frame);
    end
    "Trial Number: " + num2str(trialCount) + ", Total Steps: " + num2str(cnt) + ", State: " + log  
end
close(writerObj)
%% Q5-2
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
catPos = [8.5,3.5];

% learning
nTrials = 10;
nActs = 4;
eta = 1;
gamma = 1;
lambda = 0;
uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
uniformProbs(1,:,:) = 1/(nActs-1);
uniformProbs(1,:,3) = 0;
uniformProbs(mapSize,:,:) = 1/(nActs-1);
uniformProbs(mapSize,:,1) = 0;
uniformProbs(:,1,:) = 1/(nActs-1);
uniformProbs(:,1,2) = 0;
uniformProbs(:,mapSize,:) = 1/(nActs-1);
uniformProbs(:,mapSize,4) = 0;
uniformProbs(1,1,:) = 1/(nActs-2);
uniformProbs(1,1,2:3) = 0;
uniformProbs(1,mapSize,:) = 1/(nActs-2);
uniformProbs(1,mapSize,3:4) = 0;
uniformProbs(mapSize,1,:) = 1/(nActs-2);
uniformProbs(mapSize,1,1:2) = 0;
uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
uniformProbs(mapSize,mapSize,1:3:4) = 0;
actProbs = uniformProbs;
mapValues = zeros(mapSize,mapSize);
reward = zeros(mapSize,mapSize);
reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
m = zeros(size(actProbs));

for trialCount = 1 : nTrials
    visitedLocs = [];
    ratPos = randi(mapSize,1,2);
    flag = 0;
    cnt = 0;
    while(1)
        visitedLocs = [visitedLocs;ratPos];
        if (trialCount == nTrials)
            subplot(2,6,[1,2,3,7,8,9])
            mapGenerator(mapSize,targetPos,catPos,ratPos-0.5);
        end
        if(ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2))
            flag = 1;
            if (trialCount == nTrials)
                text(floor(mapSize/4),floor(3/4*mapSize),'!Reward Catched!','Color','red','Interpreter','latex','FontSize',12);
                log = "Reward";
                pause(0.5)
            end
        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
            flag = 1;
            if (trialCount == nTrials)
                text(floor(1*mapSize/3),floor(3/4*mapSize),'!Attacked!','Color','red','Interpreter','latex','FontSize',12);
                log = "Attacked";
                pause(0.5)
            end
        end
        if (trialCount == nTrials)
            ylabel("$$\lambda$$ = " + num2str(lambda),'interpreter','latex')
            title("Main Map at Trial = "+ num2str(trialCount) +", Step = " + num2str(cnt),'Interpreter','latex');
            subplot(2,6,[5,6])
            contourf((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
            title("Value Contours",'Interpreter','latex');
            colorbar()
            colormap("gray")

            subplot(2,6,[11,12])
            [px,py] = gradient(mapValues');  
            contour((1:mapSize)-0.5,(1:mapSize)-0.5,mapValues')
            hold on
            quiver((1:mapSize)-0.5,(1:mapSize)-0.5,px,py)
            hold off
            title("Gradient",'Interpreter','latex');
        end
        mapValuesBefore = mapValues;
        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
        mapValues = TDLamdaUPdator(mapValuesBefore,mapValues,visitedLocs,ratPos,lambda);
        actApproval = 0;
        while(actApproval == 0)
            actApproval = 1;
            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                actApproval = 0;
            end
        end

        if (cnt ~= 0)
            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
            for Action = [1,2,3,4]
                if(Action ~= actPre)
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                else
                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                end
            end
            m = edgeHandler(m,mapSize);
            softmax_func = @(x) exp(0.1*x)/sum(exp(0.1*x));
            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
        end

        if(flag == 1)
            break;
        end
        ratPosPre = ratPos;
        actPre = act;
        ratPos = ratPos + [acti,actj];
        cnt = cnt + 1;
    end 
end

%% Q5-3
clear all
close all
clc
 
mapSize = 15;
targetPos = [5.5,7.5];
catPos = [8.5,3.5];

% learning
nTrials = 500;
nActs = 4;
lambdas = [0,0.1,0.3,0.5];
etas = [0.001,0.01,0.1,1];
gammas = [0.001,0.01,0.1,1];
for k = 1 : 4
    lambda = lambdas(k);
    HitMap = zeros(length(etas),length(gammas));
    AverageHitMap = HitMap;
    for runs = 1 : 10
        runs
        cnt1 = 1;
        for gamma = gammas
            cnt2 = 1;
            for eta = etas
                uniformProbs = (1/nActs)*ones(mapSize,mapSize,nActs);
                uniformProbs(1,:,:) = 1/(nActs-1);
                uniformProbs(1,:,3) = 0;
                uniformProbs(mapSize,:,:) = 1/(nActs-1);
                uniformProbs(mapSize,:,1) = 0;
                uniformProbs(:,1,:) = 1/(nActs-1);
                uniformProbs(:,1,2) = 0;
                uniformProbs(:,mapSize,:) = 1/(nActs-1);
                uniformProbs(:,mapSize,4) = 0;
                uniformProbs(1,1,:) = 1/(nActs-2);
                uniformProbs(1,1,2:3) = 0;
                uniformProbs(1,mapSize,:) = 1/(nActs-2);
                uniformProbs(1,mapSize,3:4) = 0;
                uniformProbs(mapSize,1,:) = 1/(nActs-2);
                uniformProbs(mapSize,1,1:2) = 0;
                uniformProbs(mapSize,mapSize,:) = 1/(nActs-2);
                uniformProbs(mapSize,mapSize,1:3:4) = 0;
                actProbs = uniformProbs;
                mapValues = zeros(mapSize,mapSize);
                reward = zeros(mapSize,mapSize);
                reward(catPos(1)+0.5,catPos(2)+0.5) = -1;
                reward(targetPos(1)+0.5,targetPos(2)+0.5) = 1;
                m = zeros(size(actProbs));
                steps = [];
                for trialCount = 1 : nTrials
                    ratPos = randi(mapSize,1,2);
                    visitedLocs = [];
                    flag = 0;
                    rflag = 0;
                    cnt = 0;
                    while(1)
                        visitedLocs = [visitedLocs;ratPos];
                        if(ratPos(1)-0.5 == targetPos(1) && ratPos(2)-0.5 == targetPos(2))
                            flag = 1;
                            rflag = 1;
                        elseif(ratPos(1)-0.5 == catPos(1) && ratPos(2)-0.5 == catPos(2))
                            flag = 1;
                        end
                        mapValuesBefore = mapValues;
                        mapValues = updateValue(mapValues,uniformProbs,actProbs,ratPos,gamma,reward);
                        mapValues = TDLamdaUPdator(mapValuesBefore,mapValues,visitedLocs,ratPos,lambda);
                        actApproval = 0;
                        while(actApproval == 0)
                            actApproval = 1;
                            [act,acti,actj] = takeAction(actProbs(ratPos(1),ratPos(2),[1,2,3,4]),[1,2,3,4]);
                            if ( (ratPos(1) + acti > mapSize) || (ratPos(1) + acti < 1) || (ratPos(2) + actj > mapSize) || (ratPos(2) + actj < 1) )
                                actApproval = 0;
                            end
                        end
                    
                        if (cnt ~= 0)
                            valuePre = mapValues(ratPosPre(1),ratPosPre(2));
                            delta = mapValues(ratPos(1),ratPos(2)) - valuePre;
                            for Action = [1,2,3,4]
                                if(Action ~= actPre)
                                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) - eta*(actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                                else
                                    m(ratPosPre(1),ratPosPre(2),Action) = m(ratPosPre(1),ratPosPre(2),Action) + eta*(1-actProbs(ratPosPre(1),ratPosPre(2),Action))*delta;
                
                                end
                            end
                            m = edgeHandler(m,mapSize);
                            softmax_func = @(x) exp(x)/sum(exp(x));
                            actProbs(ratPosPre(1),ratPosPre(2),:) = softmax_func(m(ratPosPre(1),ratPosPre(2),:));
                        end
                
                        if(flag == 1)
                            break;
                        end
                        ratPosPre = ratPos;
                        actPre = act;
                        ratPos = ratPos + [acti,actj];
                        cnt = cnt + 1;
                    end
                    if (rflag == 1)
                        steps = [steps,cnt-1];
                    end
                end
                HitMap(cnt1,cnt2) = mean(steps(end-30:end));
                cnt2 = cnt2 + 1;
            end 
            cnt1 = cnt1 + 1;
        end
        AverageHitMap = AverageHitMap + HitMap;
    end
    AverageHitMap = AverageHitMap/runs;
    subplot(2,2,k)
    set(gcf,'color','w');
    imagesc(log10(gammas),log10(etas),AverageHitMap);
    title("HeatMap (lambda = "+num2str(lambda)+")",'Interpreter','latex')
    ylabel('$$\gamma$$(Logarithmic)','Interpreter','latex')
    xlabel('$$\epsilon$$(Logarithmic)','Interpreter','latex')
    c = colorbar;
    c.Label.String = 'Average Test Steps';
    colormap("gray")
end