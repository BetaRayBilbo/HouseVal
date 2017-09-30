function J=cost(params,y,x)

m=length(y);
J=(x*params'-y)'*(x*params'-y)/(2*m);

