clear all;
%close all;


out_abs_grad={};
out_abs_grad_candidate={};
out_abs_grad{1}=[];
out_abs_grad_candidate{1}=[];

out_active_inx={};
out_gamma=[];
out_X_nnz=[];
out_L_nnz=[];
out_W_nnz=[];
out_obj=[];
out_time=[];
out_ll=[];

run('results_0.m');

n = length(out_abs_grad);
p = size(out_abs_grad{1},1);

nnzpr_abs_grad=[];
nnzpr_abs_grad_candidate=[];


figure;
for i=1:n
    
    subplot(1,n,i)

    nnzpr_abs_grad(end+1) = nnz(out_abs_grad{i})/p;
    nnzpr_abs_grad_candidate(end+1) = nnz(out_abs_grad_candidate{i})/p; 

    %spy(out_abs_grad{i},'k'); hold on;
    %spy(out_abs_grad_candidate{i},'r'); 
    spy(out_active_inx{i},'k'); 

    %spy(out_abs_grad_candidate{i},'k'); 


    
end

figure
for i=1:n
    subplot(2,n,i)
    histogram(nonzeros(out_abs_grad{i}),'Normalization','cumcount','FaceColor','k');
    subplot(2,n,i+n)
    histogram(nonzeros(out_abs_grad_candidate{i}),'Normalization','cumcount','FaceColor','r');
end

figure
area([nnzpr_abs_grad;nnzpr_abs_grad_candidate]');
xticks((1:n))
legend({'Existing','Candidate'})

