clear, clc

A=csvread('train2.csv',1,0); % reads csv file skips first row
Atest=csvread('test2.csv',1,0);

B=A(:,1); B(:,2)=A(:,2)+A(:,3); B(:,3)=A(:,4);% Creates new matrix where 1st and 2nd sqft are added together
Btest=Atest(:,1); Btest(:,2)=Atest(:,2)+Atest(:,3);

xmax=max(B(:,2)); xmin=min(B(:,2));
xtstmax=max(Btest(:,2)); xtstmin=min(Btest(:,2));
% Feature scale sqft
xscale=(B(:,2)-xmin)/(xmax-xmin);
xtstscale=(Btest(:,2)-xtstmin)/(xtstmax-xtstmin);

figure;
scatter(xscale,B(:,3),'x'); hold on; % shows scaled values

p=polyfit(xscale,B(:,3),1); % finds slope and intercept of data set
ftrain=polyval(p,xscale); % creates linear curve based on above slope and intercept

plot(xscale,ftrain,'-'); hold off; % scaled data with linear fit

testout=p(1)*xtstscale+p(2);% getting price values from fit slope and intercept

plot(xtstscale,testout);

result(:,1)=Btest(:,1); result(:,2)=testout; % creates matrix with test id values and found house prices

csvwrite('result1.csv',result); % writes csv file from result matrix (added 