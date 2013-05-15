function plot_PR_iciap()
%%Function to plot P/R change according to neighbor changes
% for 3 datasets
%  for iciap experiments

X1 = [10, 25, 40, 80, 100, 200];
X2 = [10, 25, 40, 100, 200];

dataset_type = 1; %corel5k

if(dataset_type == 1) 
   %% plot P
    p_jec = [22.7, 23.8, 25.5, 26.3, 26.8];
    p_pro = [38.6, 39.5, 37.8, 36.7, 32.3];
    r_jec = [23.9, 26.7, 30.2, 30.4, 32];
    r_pro = [43.3, 44.7, 42.1, 39.3, 34.6];
    
    figure(1); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, p_jec, '-bo', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_jec, '-.bd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, p_pro, '-ro', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_pro, '-.rd', 'LineWidth', lw, 'MarkerSize', ms);
    
    xlim([0, 200]);
    ylim([20, 45]);
    
    set(gca,'XTick',[10, 40, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','100','200'});
    set(gca, 'YTick',[20, 25, 30, 35, 40, 45]);
    set(gca, 'YTickLabel', {'.20','.25', '.30', '.35', '.40', '.45'});
    set(gca, 'FontSize',14);
    
    xlabel('K nearest neighbors', 'FontSize', 13);
    ylabel('Precision and Recall', 'FontSize', 13);
    
    h = legend('P (JEC [1])', 'R (JEC [1])', 'P (Proposed)', 'R (Proposed)', 'Location', 'NorthEast');
    set(h, 'FontSize', 13);
    hold off;   
    
elseif(dataset_type ==  2) %espgame  
    %% plot P and R
    p_jec = [16.3, 18.5, 19.8, 22.6, 23.1];
    p_pro = [28.1, 32.4, 34.5, 41.6, 44.1, 39.3];
    r_jec = [17.3, 19.1, 20.4, 24.1, 25.2];
    r_pro = [19.3, 21.6, 23.9, 25.5, 26.2, 24.3];
    
    figure(1); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, p_jec, '-bo', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_jec, '-.bd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, p_pro, '-ro', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, r_pro, '-.rd', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([15, 50]);
    
    set(gca,'XTick',[10, 40, 80, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','80','100','200'});
    set(gca, 'YTick',[15, 25, 35, 45, 50]);
    set(gca, 'YTickLabel', {'.15','.25', '.35', '.45', '.50'});
    set(gca, 'FontSize',14);
    xlabel('K nearest neighbors', 'FontSize', 13);
    ylabel('Precision and Recall', 'FontSize', 13);
    
    h = legend('P (JEC [1])', 'R (JEC [1])', 'P (Proposed)', 'R (Proposed)', 'Location', 'SouthEast');
    set(h, 'FontSize', 13);
    hold off;
elseif(dataset_type == 3) %iaprtc12
    %% plot P and R
%     p_jec = [18.7, 19.6, 24.3 ,27.7, 28.4];
    p_jec = [21.7, 22.6, 25.3 ,27.7, 28.4];
    p_pro = [32.5, 41.3, 44.2, 47.6, 45.1, 36.7];
    r_jec = [20.5, 21.6, 26.1, 27.9, 29.3];
    r_pro = [28.7, 30.4, 33.2, 36.1, 34.4, 28.7];
    
    figure(1); hold on;
    grid on;
    ms = 7;
    lw = 2.5;
    
    plot(X2, p_jec, '-bo', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X2, r_jec, '-.bd', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, p_pro, '-ro', 'LineWidth', lw, 'MarkerSize', ms);
    plot(X1, r_pro, '-.rd', 'LineWidth', lw, 'MarkerSize', ms);
    xlim([0, 200]);
    ylim([15, 50]);
    
    set(gca,'XTick',[10, 40, 80, 100, 200]);
    set(gca, 'XTickLabel',{'10','40','80','100','200'});
    set(gca, 'YTick',[15, 25, 35, 45, 50]);
    set(gca, 'YTickLabel', {'.15','.25', '.35', '.45', '.50'});
    set(gca, 'FontSize',14);
    xlabel('K nearest neighbors', 'FontSize', 13);
    ylabel('Precision and Recall', 'FontSize', 13);
    
    h = legend('P (JEC [1])', 'R (JEC [1])', 'P (Proposed)', 'R (Proposed)', 'Location', 'SouthEast');
    set(h, 'FontSize', 13);
    hold off;
end
