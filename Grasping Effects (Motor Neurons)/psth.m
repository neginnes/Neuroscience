function MAT = psth(data,trials_events_matrix,window,Fs,neuron_number,figure_flag)
    Colors = [[1 0 0];[0.9290 0.6940 0.1250];[0 1 0];[0 0 1];[0 1 1];[1 0 1];[1 1 0];[0.4660 0.6740 0.1880]];
    true_labels=[' TS-ON ';' WS-ON ';'CUE-ON ';'CUE-OFF';' GO-ON ';'  SR   ';' RW-ON ';' STOP  '];
    n = floor(length(data(1,:))/window);
    MAT = arrayfun(@(x) Window(data,window,x),1:length(data(:,1)),'UniformOutput',false);
    MAT=MAT';
    MAT = cell2mat(MAT);
    MAT = mean(MAT,1);
    y=smooth(MAT*Fs/window);
    if (figure_flag == 1)
        plot((1:n)*window/Fs,y);
        Y_MAX = 1.1*max(y);
        ylim([0,Y_MAX]);
        bias = trials_events_matrix(:,1)*ones(1,size(trials_events_matrix,2));
        trials_events_matrix = trials_events_matrix - bias;
        label_times = mean(trials_events_matrix);
        hold on 
        for t = 1:size(trials_events_matrix,2)
            p(t) = plot([label_times(t)/Fs,label_times(t)/Fs],[0,Y_MAX],'color',Colors(t,:));
            hold on
        end
        legend(p,{true_labels(1:size(trials_events_matrix,2),:)},'FontSize',10,'Location','eastoutside','interpreter','latex')
        xlabel('t(s)','interpreter','latex');
        ylabel('PSTH','interpreter','latex');
        title(["neuron number  = " , num2str(neuron_number)],'interpreter','latex');
    end
end
 
 
