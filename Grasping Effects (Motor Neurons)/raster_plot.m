function raster_plot(spikeMat,Fs,trials_events_matrix,neuron_number)
    p = [];
    figure()
    Colors = [[1 0 0];[0.9290 0.6940 0.1250];[0 1 0];[0 0 1];[0 1 1];[1 0 1];[1 1 0];[0.4660 0.6740 0.1880]];
    true_labels=[' TS-ON ';' WS-ON ';'CUE-ON ';'CUE-OFF';' GO-ON ';'  SR   ';' RW-ON ';' STOP  '];
    number_of_trials = size(spikeMat,1);
    number_of_samples = size(spikeMat,2);
    time = 0 : 1/Fs : number_of_samples/Fs-1/Fs;
    for i = 1:number_of_trials 
        spike_times = time(spikeMat(i,:)==1);
        for j = 1:length(spike_times)
            plot([spike_times(j),spike_times(j)],[i-0.4,i+0.4],'k');
            hold on
        end
        for t = 1:size(trials_events_matrix,2)
           if (i==1)
                p(t) = plot([(trials_events_matrix(i,t)-trials_events_matrix(i,1))/Fs,(trials_events_matrix(i,t)-trials_events_matrix(i,1))/Fs],[(number_of_trials+1)*(i-1)/number_of_trials,(number_of_trials+1)*(i)/number_of_trials],'color',Colors(t,:));
           end
           plot([(trials_events_matrix(i,t)-trials_events_matrix(i,1))/Fs,(trials_events_matrix(i,t)-trials_events_matrix(i,1))/Fs],[(number_of_trials+1)*(i-1)/number_of_trials,(number_of_trials+1)*(i)/number_of_trials],'color',Colors(t,:));
           hold on
        end
    end
    
    legend(p,{true_labels(1:size(trials_events_matrix,2),:)},'FontSize',10,'Location','eastoutside','interpreter','latex')
    ylim([0,number_of_trials+1]);
    hold on
    ylabel('Raster Plot','interpreter','latex');
    xlabel('time(s)','interpreter','latex');
    title(["neuron number  = " , num2str(neuron_number)],'interpreter','latex');
end

