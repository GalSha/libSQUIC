

p=2000;

figure;
subplot(2,1,1)
[grad_mean,obj,nnzX,nnzL,nnzW,time] =make_res(strcat('p',num2str(p),'_a0.m') );
semilogx(cumsum(time),obj,'-*r');hold on;
[grad_mean,obj,nnzX,nnzL,nnzW,time] =make_res(strcat('p',num2str(p),'_a05.m') );
semilogx(cumsum(time),obj,'-*b');hold on;
[grad_mean,obj,nnzX,nnzL,nnzW,time] =make_res(strcat('p',num2str(p),'_a1.m') );
semilogx(cumsum(time),obj,'-*g');hold on;
legend({'a=0.0','a=0.5','a=1.0'});
xlabel('Total Runtime [Sec]','interpreter','latex')
ylabel('Objective Value','interpreter','latex')
subplot(2,1,2)
[grad_mean,obj,nnzX,nnzL,nnzW,time] =make_res(strcat('p',num2str(p),'_a0.m') );
semilogy(nnzX,'-*r');hold on;
[grad_mean,obj,nnzX,nnzL,nnzW,time] =make_res(strcat('p',num2str(p),'_a05.m') );
semilogy(nnzX,'-*b');hold on;
[grad_mean,obj,nnzX,nnzL,nnzW,time] =make_res(strcat('p',num2str(p),'_a1.m') );
semilogy(nnzX,'-*g');hold on;
legend({'a=0.0','a=0.5','a=1.0'});
xlabel('Iterations','interpreter','latex')
ylabel('$nnz(\mathbf{\Theta}$)/$p$','interpreter','latex')
