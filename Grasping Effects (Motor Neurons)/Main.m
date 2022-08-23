%% Negin Esmaeil Zadeh , Samin Beheshti, Fatemeh Salehin
%% Data loading
clear all
close all
clc
load('i140703-001_lfp-spikes.mat') 
Fs = block.segments{1,1}.spiketrains{1,1}.sampling_rate;
%% Pre procossenig
clc
close all
Fs = block.segments{1,1}.spiketrains{1,1}.sampling_rate;
an_trial_event_labels  = block.segments{1,1}.events{1,1}.an_trial_event_labels;
Time = block.segments{1,1}.events{1,1}.times;
number_of_trials = 0;
TS_ON_index_vector =[];
STOP_index_vector = [];
SR_index_vector = [];
SR_trials_vector = [];
GO_ON_index_vector = [];
GO_ON_trials_vector = [];
for i = 1 : size(an_trial_event_labels,1)
    if (an_trial_event_labels(i,:) ==       'TS-ON      ')
        TS_ON_index_vector = [TS_ON_index_vector;i];
        number_of_trials = number_of_trials + 1;
    end
    if (an_trial_event_labels(i,:) == 'STOP       ')
        STOP_index_vector = [STOP_index_vector;i];
    end
    if (an_trial_event_labels(i,:) == 'SR         ') 
        SR_index_vector = [SR_index_vector;i];
        SR_trials_vector = [SR_trials_vector;number_of_trials];
    end
    if (an_trial_event_labels(i,:) == 'GO-ON      ')
        GO_ON_index_vector = [GO_ON_index_vector;i];
        GO_ON_trials_vector = [GO_ON_trials_vector;number_of_trials];
    end
end
%% raster plot
clc
close all
neurons = block.segments{1,1}.spiketrains;
N = 20;
neuron_number = 1;
Number_of_plotting_neurons = size(neurons,2);
for i = 1 : size(neurons,2)
    random_trials_matrix = [];
    random_trials_events_matrix = [];
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    spike_times = block.segments{1,1}.spiketrains{1,i}.times;
    r = zeros(1,N);
    for l = 1 : N
        error = 1;
        while(error == 1)
            randm = randperm(number_of_trials,1);
            if (an_trial_event_labels(STOP_index_vector(randm)- 1 , :) ==     'RW-ON      ')
                r(l) = randm;
                error = 0;
            end
        end
    end
    r = sort(r);

    ranges =  Time(STOP_index_vector(r)) - Time(TS_ON_index_vector(r))  ; 
    M = max(ranges);
    random_trials_matrix = zeros (N,M);
    true_labels=['TS-ON      ';'WS-ON      ';'CUE-ON     ';'CUE-OFF    ';'GO-ON      ';'SR         ';'RW-ON      ';'STOP       '];
    for j = 1 : length(r)
        rj = r(j);
        trial_stop =  Time(STOP_index_vector(rj));
        trial_start = Time(TS_ON_index_vector(rj));
        flag = 0;
        for k = 1:size(spike_times,2)
            if (spike_times(k)>= trial_start)
                flag = 1;
            end
            if (spike_times(k)> trial_stop)
                flag = 0;
            end
            if (flag == 1)
                random_trials_matrix(j,spike_times(k)-trial_start+1) = 1;
                c = 0;
                for t = 1:size(true_labels,1)
                    while (true_labels(t,:)~=an_trial_event_labels(TS_ON_index_vector(rj)+c,:))
                        c = c + 1;
                    end
                    random_trials_events_matrix(j,t)=Time(TS_ON_index_vector(rj)+c);
                    c = c + 1;
                end

            end
        end 
    end
   raster_plot(random_trials_matrix,Fs,random_trials_events_matrix,neuron_number)
   neuron_number = neuron_number + 1;
end
 

%% PSTH
neurons = block.segments{1,1}.spiketrains;
neuron_number = 1;
Number_of_plotting_neurons = size(neurons,2);
for i = 1 : 2 :size(neurons,2)
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    spike_times = block.segments{1,1}.spiketrains{1,i}.times;
    ranges =  Time(STOP_index_vector) - Time(TS_ON_index_vector); 
    M = max(ranges);
    trials_matrix = zeros(length(TS_ON_index_vector),M);
    true_labels=['TS-ON      ';'WS-ON      ';'CUE-ON     ';'CUE-OFF    ';'GO-ON      ';'SR         ';'RW-ON      ';'STOP       '];
    for j = 1 : length(TS_ON_index_vector)
        trial_stop =  Time(STOP_index_vector(j));
        trial_start = Time(TS_ON_index_vector(j));
        flag = 0;
        for k = 1:size(spike_times,2)
            if (spike_times(k)>= trial_start)
                flag = 1;
            end
            if (spike_times(k)> trial_stop)
                flag = 0;
            end
            if (flag == 1)
                trials_matrix(j,spike_times(k)-trial_start+1) = 1;
                c = 0;
                for t = 1:size(true_labels,1)
                    while (true_labels(t,:)~=an_trial_event_labels(TS_ON_index_vector(j)+c,:))
                        c = c + 1;
                    end
                    trials_events_matrix(j,t)=Time(TS_ON_index_vector(j)+c);
                    c = c + 1;
                end
            end
        end 
    end
   window_width = 160;
   figure_flag = 1;
   figure();
   p = psth (trials_matrix, trials_events_matrix, window_width,Fs,neuron_number,figure_flag); 
   neuron_number = neuron_number + i;
end
%% ISI
neurons = block.segments{1,1}.spiketrains;
neuron_number = 1;
Number_of_plotting_neurons = size(neurons,2);
for i = 2 : size(neurons,2)
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    spike_times = block.segments{1,1}.spiketrains{1,i}.times;
    figure()
    spike_intervals = (spike_times(2:end) - spike_times(1:end-1))/Fs; 
    h = histogram(spike_intervals,'Normalization','probability');
    xlabel('Time Interval(s)','interpreter','latex');
    ylabel('Probability','interpreter','latex');
    title(["ISI for neuron number  = " , num2str(neuron_number)],'interpreter','latex');
    figure()
    histogram(spike_intervals, 3000*(1+floor(min(spike_intervals))),'Normalization','probability')
    xlabel('Time Interval(s)','interpreter','latex');
    ylabel('Probability','interpreter','latex');
    title([" ISI for neuron number  = " , num2str(neuron_number)],'interpreter','latex');
    figure()
    hi = histfit(spike_intervals,h.NumBins,'Exponential','Normalization','probability')
    hi(2).YData = hi(2).YData/max(hi(2).YData);
    hi(1).YData = hi(1).YData/max(hi(1).YData);
    xlabel('Time Interval(s)','interpreter','latex');
    ylabel('Probability','interpreter','latex');
    title(["ISI for neuron number  = " , num2str(neuron_number)],'interpreter','latex');
    neuron_number = neuron_number + 1;
    
end


    
%% permutation SR
neurons = block.segments{1,1}.spiketrains;
neuron_number = 1;
Number_of_plotting_neurons = size(neurons,2);
for i = 1 : size(neurons,2)
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    spike_times = block.segments{1,1}.spiketrains{1,i}.times;
    SR_Time =  Time(SR_index_vector) ; 
    SR_Ranges = (1*Fs + SR_Time) - (SR_Time - 1*Fs);
    SR_matrix = zeros(length(SR_index_vector),SR_Ranges(1));

    true_labels=['TS-ON      ';'WS-ON      ';'CUE-ON     ';'CUE-OFF    ';'GO-ON      ';'SR         ';'RW-ON      ';'STOP       '];
    for j = 1 : length(SR_index_vector)
        SR_stop =  Time(SR_index_vector(j)) + 1*Fs;
        SR_start = Time(SR_index_vector(j)) - 1*Fs;
        flag = 0;
        for k = 1:size(spike_times,2)
            if (spike_times(k)>= SR_start)
                flag = 1;
            end
            if (spike_times(k)> SR_stop)
                flag = 0;
            end
            if (flag == 1)
                SR_matrix(j,spike_times(k)-SR_start+1) = 1;
            end
        end 
    end
   window_width = 160;
   SR_events_matrix =[];
   figure_flag = 0;
   SR_PSTH(i,:) = psth (SR_matrix, SR_events_matrix, window_width,Fs,neuron_number,figure_flag); 
   neuron_number = neuron_number + 1;
end


%% 
AVG_SR_PSTHs = mean(SR_PSTH);
C = AVG_SR_PSTHs;
Before_SR_event = AVG_SR_PSTHs(:,1:floor(size(AVG_SR_PSTHs,2)/2));
After_SR_event = AVG_SR_PSTHs(:,1+floor(size(AVG_SR_PSTHs,2)/2):end);
N1 = floor(size(AVG_SR_PSTHs,2)/2);
N2 = floor(size(AVG_SR_PSTHs,2)/2)+1;
M = 100000;
delta_psth_distribution = zeros(1,M);
for i = 1 : M
    C = AVG_SR_PSTHs;
    R1 = randperm(size(AVG_SR_PSTHs,2),N1);
    C(R1) = -1;
    R2 = find(C>=0);
    delta_psth_distribution(i) = mean(AVG_SR_PSTHs(R1)) - mean(AVG_SR_PSTHs(R2));
end
delta_psth_star = mean(After_SR_event) - mean(Before_SR_event)
figure ()
g = histogram(delta_psth_distribution,'Normalization','probability')
xlabel('Delta psth' , 'interpreter','latex')
ylabel('Probability' , 'interpreter','latex')
title('Distribution of first null hypothesis' , 'interpreter','latex')
hold on
plot ([delta_psth_star,delta_psth_star] , [0 , 1.1 * max(g.Values)] ,'color' , 'r')

[h,p] = ttest(AVG_SR_PSTHs)

%% permutation GO-ON
neurons = block.segments{1,1}.spiketrains;
neuron_number = 1;
Number_of_plotting_neurons = size(neurons,2);
for i = 1 : size(neurons,2)
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    spike_times = block.segments{1,1}.spiketrains{1,i}.times;
    GO_ON_Time =  Time(GO_ON_index_vector) ; 
    GO_ON_Ranges = (0.05*Fs + GO_ON_Time) - (GO_ON_Time - 0.05*Fs);
    GO_ON_matrix = zeros(length(GO_ON_index_vector),GO_ON_Ranges(1));

    true_labels=['TS-ON      ';'WS-ON      ';'CUE-ON     ';'CUE-OFF    ';'GO-ON      ';'SR         ';'RW-ON      ';'STOP       '];
    for j = 1 : length(GO_ON_index_vector)
        GO_ON_stop =  Time(GO_ON_index_vector(j)) + 0.05*Fs;
        GO_ON_start = Time(GO_ON_index_vector(j)) - 0.05*Fs;
        flag = 0;
        for k = 1:size(spike_times,2)
            if (spike_times(k)>= GO_ON_start)
                flag = 1;
            end
            if (spike_times(k)> GO_ON_stop)
                flag = 0;
            end
            if (flag == 1)
                GO_ON_matrix(j,spike_times(k)-GO_ON_start+1) = 1;
            end
        end 
    end
   window_width = 160;
   GO_ON_events_matrix =[];
   figure_flag = 0;
   GO_ON_PSTH(i,:) = psth (GO_ON_matrix, GO_ON_events_matrix, window_width,Fs,neuron_number,figure_flag); 
   neuron_number = neuron_number + 1;
end
%%
AVG_GO_ON_PSTHs = mean(GO_ON_PSTH);
C = AVG_GO_ON_PSTHs;
Before_GO_ON_event = AVG_GO_ON_PSTHs(:,1:floor(size(AVG_GO_ON_PSTHs,2)/2));
After_GO_ON_event = AVG_GO_ON_PSTHs(:,1+floor(size(AVG_GO_ON_PSTHs,2)/2):end);
N1 = floor(size(AVG_GO_ON_PSTHs,2)/2);
N2 = floor(size(AVG_GO_ON_PSTHs,2)/2)+1;
M = 100000;
delta_psth_distribution = zeros(1,M);
for i = 1 : M
    C = AVG_GO_ON_PSTHs;
    R1 = randperm(size(AVG_GO_ON_PSTHs,2),N1);
    C(R1) = -1;
    R2 = find(C>=0);
    delta_psth_distribution(i) = mean(AVG_GO_ON_PSTHs(R1)) - mean(AVG_GO_ON_PSTHs(R2));
end
delta_psth_star = mean(After_GO_ON_event) - mean(Before_GO_ON_event)
figure()
h_total = histogram(delta_psth_distribution,'Normalization','probability');
area_total=sum(h_total.Values)*h_total.BinWidth
xlabel('Delta psth' , 'interpreter','latex')
ylabel('Probability' , 'interpreter','latex')
title('Distribution of second null hypothesis' , 'interpreter','latex')
hold on
plot ([delta_psth_star,delta_psth_star] , [0 , 1.1 * max(h_total.Values)] ,'color' , 'r')
figure()
h_star = histogram(delta_psth_distribution,'BinLimits',[delta_psth_star , max(delta_psth_distribution)],'Normalization','probability');
area_star=sum(h_star.Values)*h_star.BinWidth
p_value = area_star/area_total
[h,p] = ttest(AVG_GO_ON_PSTHs)

%% Fano Factor
neurons = block.segments{1,1}.spiketrains;
neuron_number = 1;
Fano_Factors_before = [];
Fano_Factors_after = [];
Number_of_plotting_neurons = size(neurons,2);
for i = 1 : size(neurons,2)
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    spike_times = block.segments{1,1}.spiketrains{1,i}.times;
    SR_Time =  Time(SR_index_vector) ; 
    SR_Ranges = (0.3*Fs + SR_Time) - (SR_Time - 0.3*Fs);
    SR_matrix = zeros(length(SR_index_vector),SR_Ranges(1));

    true_labels=['TS-ON      ';'WS-ON      ';'CUE-ON     ';'CUE-OFF    ';'GO-ON      ';'SR         ';'RW-ON      ';'STOP       '];
    for j = 1 : length(SR_index_vector)
        SR_stop =  Time(SR_index_vector(j)) + 0.3*Fs;
        SR_start = Time(SR_index_vector(j)) - 0.3*Fs;
        flag = 0;
        for k = 1:size(spike_times,2)
            if (spike_times(k)>= SR_start)
                flag = 1;
            end
            if (spike_times(k)> SR_stop)
                flag = 0;
            end
            if (flag == 1)
                SR_matrix(j,spike_times(k)-SR_start+1) = 1;
            end
        end 
    end
    SR_matrix_before = SR_matrix(:,1:floor(size(SR_matrix,2)/2));
    SR_matrix_after = SR_matrix(:,1+floor(size(SR_matrix,2)/2):end);

    Fano_Factors_before = [Fano_Factors_before;var(SR_matrix_before,0,2)./(mean(SR_matrix_before,2))];
    Fano_Factors_after = [Fano_Factors_after;var(SR_matrix_after,0,2)./(mean(SR_matrix_after,2))];
    neuron_number = neuron_number + 1;
end
%%

n1 = Fano_Factors_before;
n2 = Fano_Factors_after;
n1(isnan(Fano_Factors_before)) = NaN;
n2(isnan(Fano_Factors_after)) = NaN;

histogram(rmmissing(n1),'Normalization','probability')
hold on
histogram(rmmissing(n2),'Normalization','probability')
legend("fano factos histogram before SR event ","fano factos histogram after SR event ",'FontSize',10,'Location','eastoutside','interpreter','latex')
xlabel('Fano Factor' ,'interpreter','latex')
ylabel('Probability' ,'interpreter','latex')

%% Different trial types
PG_LF = [];
SG_LF = [];
SG_HF = [];
PG_HF = [];
for i = 1 : size(TS_ON_index_vector)
    flag_PG = 0;
    flag_SG = 0;
    flag_HF = 0;
    flag_LF = 0;
    for j = TS_ON_index_vector(i) : STOP_index_vector(i)
        if (block.segments{1,1}.events{1,1}.labels(j,:) == '65365')
            flag_PG = 1;
        end
        if(block.segments{1,1}.events{1,1}.labels(j,:) == '65370')
            flag_SG = 1;
        end
        if(block.segments{1,1}.events{1,1}.labels(j,:) == '65369')
            flag_LF = 1;
        end
        if(block.segments{1,1}.events{1,1}.labels(j,:) == '65366')
            flag_HF = 1;
        end
    end
    if (flag_PG == 1 && flag_LF == 1)
        PG_LF = [PG_LF , i];

    elseif (flag_PG == 1 && flag_HF == 1)
        PG_HF = [PG_HF , i];

    elseif (flag_SG == 1 && flag_LF == 1)
        SG_LF = [SG_LF , i];

    elseif (flag_SG == 1 && flag_HF == 1)
        SG_HF = [SG_HF , i];
    end 
end
%% PSTH of different trial types
neurons = block.segments{1,1}.spiketrains;
neuron_number = 1;
Number_of_plotting_neurons = size(neurons,2);
for i = 1 : size(neurons,2)
    if (neuron_number>Number_of_plotting_neurons)
        break;
    end
    figure();
    for f = 1 : 4
        if (f == 1)
            trails_vector = PG_LF;
            STR = "PG,LF";
        elseif (f == 2)
            trails_vector = PG_HF;
            STR = "PG,HF";
        elseif (f == 3)
            trails_vector = SG_LF;
            STR = "SG,LF";
        elseif (f == 4)
            trails_vector = SG_HF;
            STR = "SG,HF";
        end
        spike_times = block.segments{1,1}.spiketrains{1,i}.times;
        ranges =  Time(STOP_index_vector) - Time(TS_ON_index_vector); 
        M = max(ranges);
        trials_matrix = zeros(length(TS_ON_index_vector),M);
        true_labels=['TS-ON      ';'WS-ON      ';'CUE-ON     ';'CUE-OFF    ';'GO-ON      ';'SR         ';'RW-ON      ';'STOP       '];
        for idx = 1 : size(trails_vector,2)
            j = trails_vector(idx);
            trial_stop =  Time(STOP_index_vector(j));
            trial_start = Time(TS_ON_index_vector(j));
            flag = 0;
            for k = 1:size(spike_times,2)
                if (spike_times(k)>= trial_start)
                    flag = 1;
                end
                if (spike_times(k)> trial_stop)
                    flag = 0;
                end
                if (flag == 1)
                    trials_matrix(j,spike_times(k)-trial_start+1) = 1;
                    c = 0;
                    for t = 1:size(true_labels,1)
                        while (true_labels(t,:)~=an_trial_event_labels(TS_ON_index_vector(j)+c,:))
                            c = c + 1;
                        end
                        trials_events_matrix(j,t)=Time(TS_ON_index_vector(j)+c);
                        c = c + 1;
                    end
                end
            end 
        end
        window_width = 160;
        figure_flag = 1;
        subplot(2,2,f)
        p = psth (trials_matrix, trials_events_matrix, window_width,Fs,neuron_number,figure_flag);
        
        ylabel(["PSTH", STR],'interpreter','latex');
    end
    neuron_number = neuron_number + 1;
end
