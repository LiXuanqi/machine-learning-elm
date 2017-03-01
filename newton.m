function [a]=newton(fun,x1,eps)
%%%%%%%%%%%%
%function:Find the min of a equation.
%fun:ObjectiveFunction
%x:the initial point
%eps:error
%%%%%%%%%%%%
syms x 
f=fun ;
grad1=jacobian(f,x); 
grad2=jacobian(grad1,x); 
k=0; 
while 1    
    grad1z=subs(subs(grad1,x,x1)); 
   grad2z=subs(subs(grad2,x,x1));
    x2=x1-inv(grad2z)*(grad1z)'; 
    if norm(x1-x2)<eps     
        break; 
    else
       
        k=k+1;  
        x1=x2; 

        
    end
end
a=x1;
end
