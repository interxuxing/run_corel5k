data1=0+randn(1,1000);
data2=10+randn(1,1000);
data3=20+randn(1,1000);
hist(data1);
hold on;
hist(data2);
hist(data3);
hold off;
h=findobj(gca,'Type','patch');
display(h)
set(h(1),'FaceColor','r','EdgeColor','k');
set(h(2),'FaceColor','g','EdgeColor','k');
set(h(3),'FaceColor','b','EdgeColor','k');