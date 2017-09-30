clear, clc

%% Setting up training using gradient descent

A=csvread('train2.csv',1,0); % reads csv file skips first row

B=A(:,1); B(:,2)=A(:,2)+A(:,3); B(:,3)=A(:,4);% Creates new matrix where 1st and 2nd sqft are added together

xmax=max(B(:,2)); xmin=min(B(:,2));
% Feature scale sqft
xscale=(B(:,2)-xmin)/(xmax-xmin);
x=[ones(1460,1),xscale]; % Creates matrix with x0 and x1
m=length(B);

param=[0,0]; learnrt=0.5;i=1;   % LEarning parameters guess and learn rate.
for j=1:3000    % guess the # of iterations
    if cost(param,B(:,3),x)>0
        param(1,1)=param(1,1)+learnrt*(1/m)*(B(:,3)-x*param')'*x(:,1);
        param(1,2)=param(1,2)+learnrt*(1/m)*(B(:,3)-x*param')'*x(:,2);
        costhistory(j)=cost(param,B(:,3),x);
        
    else
        break;
       
    end
end

h=x*param';

figure; hold on;
scatter(xscale,B(:,3),'x');
plot(xscale,h,'-');
hold off;

%% Test Fitting

Atest=csvread('test2.csv',1,0);
Btest=Atest(:,1); Btest(:,2)=Atest(:,2)+Atest(:,3);
xtstmax=max(Btest(:,2)); xtstmin=min(Btest(:,2));
xtstscale=(Btest(:,2)-xtstmin)/(xtstmax-xtstmin);
xtst=[ones(length(Btest),1),xtstscale];

htst=xtst*param';

plot(xtstscale,htst);

output=[Btest(:,1),htst];

csvwrite('GradDes1.csv',output);