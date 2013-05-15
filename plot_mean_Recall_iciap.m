function plot_mean_Recall_iciap()
%% This function is to plot a figure to show the word-specific mean recall
%   Here we compare 3 models
%   JEC, sML, Proposed

%% prepare for statistic
dataset_type = 1;

if (dataset_type == 1)
    mean_recall_jec = [0.155, 0.305, 0.403, 0.36, 0.498]';
    mean_recall_sml = [0.35, 0.59, 0.66, 0.75, 0.76]';
    mean_recall_prop = [0.39, 0.62, 0.71, 0.72, 0.74]';
    
    bar_axe = [1, 2, 3, 4, 5];
    
    figure(1), clf, hold on;
    bar([1:5], [mean_recall_jec mean_recall_sml mean_recall_prop], 'BarWidth',1.1 );
    
    ylim([0, 0.8]);
    set(gca,'XTick',[1, 2, 3, 4, 5]);
    set(gca, 'XTickLabel',{'5','10','50','100','100+'});
    set(gca, 'YTick',[0, 0.4, 0.8]);
    set(gca, 'YTickLabel', {'0','.4', '.8'});
    set(gca, 'FontSize',14);
    
    h=findobj(gca,'Type','patch');
    display(h);
    set(h(1),'FaceColor','r','EdgeColor','k');
    set(h(2),'FaceColor','y','EdgeColor','k');
    set(h(3),'FaceColor','b','EdgeColor','k');
    
    xlabel('Images per label', 'FontSize', 13);
    ylabel('Mean Recall', 'FontSize', 13);
    
    hold off;
    
elseif(dataset_type == 2)
    
else(dataset_type == 3)
    
end