load('UnitsData')

%% Step1.1
clc
close all
k = 5;
nUnits = length(Unit);
R = randperm(nUnits,k);
start_time = -1.2;
stop_time = 2;
figure_flag = 1;
window_width = 0.01;
PSTH = [];
for r = R
    data = Unit(r).Trls;
    unit_number = r;
    figure()
    set(gcf,'color','w');
    PSTH = [PSTH; psth(data,window_width,unit_number,start_time,stop_time,figure_flag)];
end


%% Step1.2
clc
close all
k = 5;
nUnits = length(Unit);
R = randperm(nUnits,k);
start_time = -1.2;
stop_time = 2;
figure_flag = 1;
window_width = 0.01;
PSTH = [];
for r = R
    figure()
    set(gcf,'color','w');
    for condition = 1 : 6
        subplot(3,2,condition)
        data = {};
        idx = Unit(r).Cnd(condition).TrialIdx;
        for i = 1 : length(idx)
            data{i,1} = Unit(r).Trls{idx(i)};
        end
        unit_number = r;
        PSTH = [PSTH; psth(data,window_width,unit_number,start_time,stop_time,figure_flag)];
        title("unit number  = " + num2str(unit_number) + ", condition  = " + num2str(condition),'interpreter','latex');
    end
end

%% Step1.3
clc
close all
nUnits = length(Unit);
k = length(Unit);
R = randperm(nUnits,k);
start_time = -1.2;
stop_time = 2;
figure_flag = 0;
window_width = 0.01;
Psth = 0;
PSTH = [];
for condition = 1 : 6 
    for r = R
        data = {};
        idx = Unit(r).Cnd(condition).TrialIdx;
        for i = 1 : length(idx)
            data{i,1} = Unit(r).Trls{idx(i)};
        end
        unit_number = r;
        Psth = Psth + psth(data,window_width,unit_number,start_time,stop_time,figure_flag);
    end
    PSTH = [PSTH ; Psth/nUnits];
end

for condition = 1 : 6
    y = smooth(PSTH(condition,:));
    subplot(3,2,condition)
    set(gcf,'color','w');
    plot(start_time+window_width/2:window_width:stop_time-window_width/2,y);
    Y_MAX = 1.1*max(y);
    ylim([0,Y_MAX]);
    xlim([start_time,stop_time]);
    hold on
    plot([0,0],[0,Y_MAX]);
    legend('PSTH','Cue Onset','interpreter','latex')
    xlabel('t(s)','interpreter','latex');
    ylabel('PSTH','interpreter','latex');
    title("condition  = " + num2str(condition),'interpreter','latex');
end

%% Step2

clc
close all
nUnits = length(Unit);
k = nUnits;
R = 1:k;
start_time = -1.2;
stop_time = 2;
figure_flag = 0;
window_width = 3.2;
PSTH = [];
Value = [3,-1;3,1;6,-1;6,1;9,-1;9,1];
X1_Pvalue = [];
X2_Pvalue = [];
for r = R
    PSTH = [];
    for i = 1 : length(Unit(r).Trls)
        data = Unit(r).Trls{i,:};
        unit_number = r;
        PSTH = [PSTH; singleTrialpsth(data,window_width,unit_number,start_time,stop_time,figure_flag)];
    end
    CndVector = zeros(length(Unit(r).Trls),2);
    for condition = 1 : 6
        idx = Unit(r).Cnd(condition).TrialIdx;
        CndVector(idx,:) = ones(length(idx),1)*Value(condition,:);
    end
    mdl = fitglm(CndVector,PSTH);
    Coefficients = table2array(mdl.Coefficients);
    X1_Pvalue = [X1_Pvalue,Coefficients(2,4)];
    X2_Pvalue = [X2_Pvalue,Coefficients(3,4)];
%     X1_Pvalue = [];
%     X2_Pvalue = [];
%     for i = 1 : size(PSTH,2)
%         mdl = fitglm(CndVector,PSTH(:,i));
%         Coefficients = table2array(mdl.Coefficients);
%         X1_Pvalue = [X1_Pvalue,Coefficients(2,4)];
%         X2_Pvalue = [X2_Pvalue,Coefficients(3,4)];
%     end
%     plot(start_time + window_width/2 :window_width:stop_time - window_width/2,X1_Pvalue);
%     hold on
%     plot(start_time + window_width/2 :window_width:stop_time - window_width/2,X2_Pvalue);
%     xlabel('t(s)','interpreter','latex');
%     ylabel('Pvlaue','interpreter','latex');
%     legend(['Reward','Location'],'interpreter','latex')
%     title("unit number  = " + num2str(unit_number),'interpreter','latex');
end
%%
set(gcf,'color','w');
subplot(2,1,1)
stem(R,X1_Pvalue);
hold on
stem(R,X2_Pvalue);
xlabel('unit number','interpreter','latex');
ylabel('Pvlaue','interpreter','latex');
legend(' Reward ',' Location ','interpreter','latex');
title('Pvalue of each unit','interpreter','latex');

subplot(2,1,2)
idx1 = find(X1_Pvalue<=0.05);
stem(R(idx1),X1_Pvalue(idx1));
hold on
idx2 = find(X2_Pvalue<=0.05);
stem(R(idx2),X2_Pvalue(idx2));
xlabel('unit number','interpreter','latex');
ylabel('Pvlaue','interpreter','latex');
legend(' Reward ',' Location ','interpreter','latex');
title('Pvalue of the units with Pvalue $$\le 0.05$$','interpreter','latex');


% %% Step2.2
% clc
% close all
% 
% nUnits = length(Unit);
% k = nUnits;
% R = 1:k;
% start_time = -1.2;
% stop_time = 2;
% figure_flag = 0;
% window_width = 0.01;
% PSTH = [];
% LocEncoderUnits = [];
% RewEncoderUnits = [];
% Value = [3,-1;3,1;6,-1;6,1;9,-1;9,1];
% for r = R
%     PSTH = [];
%     for i = 1 : length(Unit(r).Trls)
%         data = Unit(r).Trls{i,:};
%         unit_number = r;
%         PSTH = [PSTH; singleTrialpsth(data,window_width,unit_number,start_time,stop_time,figure_flag)];
%     end
%     CndVector = zeros(length(Unit(r).Trls),2);
%     for condition = 1 : 6
%         idx = Unit(r).Cnd(condition).TrialIdx;
%         CndVector(idx,:) = ones(length(idx),1)*Value(condition,:);
%     end
%     X1_Pvalue = [];
%     X2_Pvalue = [];
%     for i = 1 : size(PSTH,2)
%         mdl = fitglm(CndVector,PSTH(:,i));
%         Coefficients = table2array(mdl.Coefficients);
%         X1_Pvalue = [X1_Pvalue,Coefficients(2,4)];
%         X2_Pvalue = [X2_Pvalue,Coefficients(3,4)];
%     end
%     idx1 = find(X1_Pvalue<=0.05);
%     idx2 = find(X2_Pvalue<=0.05);
%     n1 = length(find(idx1 > 1.2/window_width));
%     n2 = length(find(idx2 > 1.2/window_width));
%     if (n1>0)
%         RewEncoderUnits = [RewEncoderUnits,r];
%     end
%     if (n2>0)
%         LocEncoderUnits = [LocEncoderUnits,r];
%     end
% end

%% Step3
clc
close all

nUnits = length(Unit);
k = nUnits;
R = 1:k;
start_time = -1.2;
stop_time = 2;
figure_flag = 0;
window_width = 0.01;
PSTH = [];
popActivity = [];
for condition = 1 : 6
    PSTH = [];
    for r = R
        data = {};
        idx = Unit(r).Cnd(condition).TrialIdx;
        for i = 1 : length(idx)
            data{i,1} = Unit(r).Trls{idx(i)};
        end
        unit_number = r;
        PSTH = [PSTH; psth(data,window_width,unit_number,start_time,stop_time,figure_flag)];
    end
    popActivity = [popActivity;dimensionReduction(PSTH,3)];
end

Colors = ['r','b','g','m','c','y'];
figure()
set(gcf,'color','w');
for condition = 1 : 6
    plot3(smooth(popActivity(3*(condition-1)+1,:)),smooth(popActivity(3*(condition-1)+2,:)),smooth(popActivity(3*(condition-1)+3,:)),'color',Colors(condition),'LineWidth',1)
    hold on
    grid on
    s1 = scatter3(popActivity(3*(condition-1)+1,1),popActivity(3*(condition-1)+2,1),popActivity(3*(condition-1)+3,1),'filled','o','MarkerFaceColor',Colors(condition));
    s1.HandleVisibility = 'off';
    hold on
    s2 = scatter3(popActivity(3*(condition-1)+1,end),popActivity(3*(condition-1)+2,end),popActivity(3*(condition-1)+3,end),80,'filled','p','MarkerFaceColor',Colors(condition));
    s2.HandleVisibility = 'off';
    hold on
end
title('population activity','interpreter','latex')
xlabel('dim 1','interpreter','latex')
ylabel('dim 2','interpreter','latex')
zlabel('dim 3','interpreter','latex')
legend('condition1','condition2','condition3','condition4','condition5','condition6','interpreter','latex');


%% Step4
clc
close all

nUnits = length(Unit);
k = nUnits;
R = 1:k;
start_time = -1.2;
stop_time = 2;
figure_flag = 0;
window_width = 0.01;
PSTH = [];
nConditions = 6;
DATA = zeros(nConditions,nUnits,(stop_time-start_time)/window_width);
for condition = 1 : nConditions
    PSTH = [];
    for r = R
        data = {};
        idx = Unit(r).Cnd(condition).TrialIdx;
        for i = 1 : length(idx)
            data{i,1} = Unit(r).Trls{idx(i)};
        end
        unit_number = r;
        PSTH = [PSTH; psth(data,window_width,unit_number,start_time,stop_time,figure_flag)];
    end
    DATA(condition,:,:) = PSTH;
end
DATA = permute(DATA,[3,2,1]);
shuffled_DATA = CFR(DATA,nConditions);
%%
nUnits = length(Unit);
k = nUnits;
R = 1:k;
start_time = -1.2;
stop_time = 2;
figure_flag = 0;
window_width = 0.01;
popActivity = [];
for condition = 1 : nConditions
    PSTH = shuffled_DATA(:,:,condition)';
    popActivity = [popActivity;dimensionReduction(PSTH,3)];
end

Colors = ['r','b','g','m','c','y'];
figure()
set(gcf,'color','w');
for condition = 1 : nconditions
    plot3(smooth(popActivity(3*(condition-1)+1,:)),smooth(popActivity(3*(condition-1)+2,:)),smooth(popActivity(3*(condition-1)+3,:)),'color',Colors(condition),'LineWidth',1)
    hold on
    grid on
    s1 = scatter3(popActivity(3*(condition-1)+1,1),popActivity(3*(condition-1)+2,1),popActivity(3*(condition-1)+3,1),'filled','o','MarkerFaceColor',Colors(condition));
    s1.HandleVisibility = 'off';
    hold on
    s2 = scatter3(popActivity(3*(condition-1)+1,end),popActivity(3*(condition-1)+2,end),popActivity(3*(condition-1)+3,end),80,'filled','p','MarkerFaceColor',Colors(condition));
    s2.HandleVisibility = 'off';
    hold on
end
title('shuffeled data population activity','interpreter','latex')
xlabel('dim 1','interpreter','latex')
ylabel('dim 2','interpreter','latex')
zlabel('dim 3','interpreter','latex')
legend('condition1','condition2','condition3','condition4','condition5','condition6','interpreter','latex');