%%Function to plot P/R change according to neighbor changes
%  for iciap experiments

X1 = [10, 25, 40, 80, 100, 200];
X2 = [10, 25, 40, 100, 200];

dataset_type = 2; %corel5k
plot_type = 'n';

if(dataset_type == 1) 
    if(strcmp(plot_type, 'p'))
    %% plot P
    p_gs = [27.7, 28.8, 31, 31.3, 31.8];
    p_jec = [22.7, 23.8, 25.5, 26.3, 26.8];
    p_tagprop_ml = [26.1, 30, 31.4, 30.7, 31];
    p_tagprop_sml = [29.5, 30.8, 32.3, 32.6, 33];
    p_pro = [38.6, 39.5, 37.8, 36.7, 32.3];
    
    figure(1); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, p_jec, '-gd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, p_tagprop_ml, '-b*', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, p_tagprop_sml, '-ks', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, p_pro, '-r^', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([20, 40]);
    
    set(gca,'XTick',[10, 40, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','100','200'});
    set(gca, 'YTick',[20, 25, 30, 35, 40]);
    set(gca, 'YTickLabel', {'.20','.25', '.30', '.35', '.40'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('P', 'FontSize', 14);
    
    h = legend('JEC [1]', 'TagProp(ML) [2]', 'TagProp(\sigmaML) [2]', 'Proposed','Location', 'SouthEast');
    set(h, 'FontSize', 14);
    hold off;

    elseif(strcmp(plot_type, 'r'))
    %% plot R
    r_gs = [28.9, 31.7, 35.2, 35.4, 37];
    r_jec = [23.9, 26.7, 30.2, 30.4, 32];
    r_tagprop_ml = [32.7, 35, 34.8, 35.7, 36.6];
    r_tagprop_sml = [34.2, 36.4, 39.7, 41.8, 42.3];
    r_pro = [43.3, 44.7, 42.1, 39.3, 34.6];
    
    figure(2); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, r_jec, '-gd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_tagprop_ml, '-b*', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_tagprop_sml, '-ks', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_pro, '-r^', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([20, 45]);
    set(gca,'XTick',[10, 40, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','100','200'});
     set(gca, 'YTick',[20, 25, 30, 35, 40, 45]);
    set(gca, 'YTickLabel', {'.20','.25', '.30', '.35', '.40','.45'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('R', 'FontSize', 14);
    
    h = legend('JEC [1]', 'TagProp(ML) [2]', 'TagProp(\sigmaML) [2]', 'Proposed','Location', 'SouthEast');
    set(h, 'FontSize', 14);
    hold off;
    
    elseif(strcmp(plot_type, 'n'))
    %%N+
    n_gs = [86, 105, 112, 130, 146];
    n_jec = [79, 98, 105, 123, 139];
    n_tagprop_ml = [75, 87, 103, 126, 146];
    n_tagprop_sml = [97, 118, 134 ,144, 160];
    n_pro = [198, 185, 167, 123, 83];
    
    figure(3); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, n_jec, '-gd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, n_tagprop_ml, '-b*', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, n_tagprop_sml, '-ks', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, n_pro, '-r^', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([50, 220]);
    set(gca,'XTick',[10, 40, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','100','200'});
    set(gca, 'YTick',[50, 100, 150, 200]);
    set(gca, 'YTickLabel', {'50', '100', '150', '200'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('N+', 'FontSize', 14);
    
    h = legend('JEC [1]', 'TagProp(ML) [2]', 'TagProp(\sigmaML) [2]', 'Proposed','Location', 'NorthEast');
    set(h, 'FontSize', 14);
    hold off;
    
    end
    
    
elseif(dataset_type ==  2)

    %% plot P
    p_jec = [18.7, 19.6, 24.3 ,27.7, 28.4];
    p_pro = [32.5, 41.3, 44.2, 47.6, 45.1, 36.7];
    r_jec = [20.5, 21.6, 26.1, 27.9, 29.3];
    r_pro = [28.7, 30.4, 33.2, 36.1, 34.4, 28.7];
    
    figure(1); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, p_jec, ':b+', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_jec, ':bd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, p_pro, '-r+', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, r_pro, '-rd', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([15, 50]);
    
    set(gca,'XTick',[10, 40, 80, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','80','100','200'});
    set(gca, 'YTick',[15, 25, 35, 45, 50]);
    set(gca, 'YTickLabel', {'.15','.25', '.35', '.45', '.50'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('P and R', 'FontSize', 14);
    
    h = legend('P (JEC [1])', 'R (JEC [1])', 'P (Proposed)', 'R (Proposed)', 'Location', 'SouthEast');
    set(h, 'FontSize', 14);
    hold off;
    

elseif(dataset_type == 3)
    if(strcmp(plot_type, 'p'))
    %% plot P
    p_jec = [18.7, 19.6, 24.3 ,27.7, 28.4];
    p_pro = [32.5, 41.3, 44.2, 47.6, 45.1, 36.7];
    
    figure(1); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, p_jec, '-gd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, p_pro, '-r^', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([15, 50]);
    
    set(gca,'XTick',[10, 40, 80, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','80','100','200'});
    set(gca, 'YTick',[15, 25, 35, 45, 50]);
    set(gca, 'YTickLabel', {'.15','.25', '.35', '.45', '.50'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('P', 'FontSize', 14);
    
    h = legend('JEC [1]', 'Proposed','Location', 'SouthEast');
    set(h, 'FontSize', 14);
    hold off;

    elseif(strcmp(plot_type, 'r'))
    %% plot R
    r_jec = [20.5, 21.6, 26.1, 27.9, 29.3];
    r_pro = [28.7, 30.4, 33.2, 36.1, 34.4, 28.7];
    
    figure(2); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, r_jec, '-gd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, r_pro, '-r^', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([15, 40]);
    set(gca,'XTick',[10, 40, 80, 100, 200]);
    set(gca, 'XTickLabel',{'10','40', '80', '100','200'});
     set(gca, 'YTick',[15, 25, 35, 40]);
    set(gca, 'YTickLabel', {'.15','.25', '.35', '.40'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('R', 'FontSize', 14);
    
    h = legend('JEC [1]', 'Proposed','Location', 'SouthEast');
    set(h, 'FontSize', 14);
    hold off;
    
    elseif(strcmp(plot_type, 'n'))
    %%N+
    n_jec = [192, 211, 235, 255, 252];
    n_pro = [216, 245, 260, 264, 249, 175];
    
    figure(3); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, n_jec, '-gd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, n_pro, '-r^', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([150, 275]);
    set(gca,'XTick',[10, 40, 80, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','80','100','200'});
    set(gca, 'YTick',[150, 200, 250, 275]);
    set(gca, 'YTickLabel', {'150', '200', '250', '275'});
    xlabel('K nearest neighbors', 'FontSize', 14);
    ylabel('N+', 'FontSize', 14);
    
    h = legend('JEC [1]', 'Proposed','Location', 'NorthEast');
    set(h, 'FontSize', 14);
    hold off;
    
    end
    
else
    return;
end