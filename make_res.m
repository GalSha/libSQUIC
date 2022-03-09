function [output_grad,output_obj,output_nnzprX,output_nnzprL,output_nnzprW,output_time] =  make_res(file_name)

output_grad=[];
output_obj=[];
output_nnzprX=[];
output_nnzprL=[];
output_nnzprW=[];
output_time=[];

run(file_name)
end