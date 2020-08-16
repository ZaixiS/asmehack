data=csvread('heatmap_50_0.5444.csv',1,0);
x=data(:,1);
y=data(:,2);
label=data(:,3);
predict=data(:,4);

x_min=min(x);
x_max=max(x);
y_min=min(y);
y_max=max(y);

pixels=100;
[X,Y]=meshgrid(0:1/pixels:1);
Z_label=zeros(size(X))+0.5;
Z_predict=zeros(size(X))+0.5;

for point=1:size(x,1)
    current_x=round((x(point)-x_min)/(x_max-x_min)*pixels)+1;
    current_y=round((y(point)-y_min)/(y_max-y_min)*pixels)+1;
    current_label=label(point);
    if current_label==0
        Z_label(current_x,current_y)=0.1;
    elseif current_label==1
        Z_label(current_x,current_y)=0.25;
    elseif current_label==2
        Z_label(current_x,current_y)=0.75;
    else
        Z_label(current_x,current_y)=1.0;
    end
    current_predict=predict(point);
    if current_predict==0
        Z_predict(current_x,current_y)=0.1;
    elseif current_predict==1
        Z_predict(current_x,current_y)=0.25;
    elseif current_predict==2
        Z_predict(current_x,current_y)=0.75;
    else
        Z_predict(current_x,current_y)=1.0;
    end
end
subplot(1,2,1);mesh(X,Y,Z_label);title('label');
subplot(1,2,2);mesh(X,Y,Z_predict);title('predict');
saveas(gcf,'heatmap.png');