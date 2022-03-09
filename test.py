import SQUIC 
import numpy as np

from sklearn.metrics import f1_score

# set location of libSQUIC (set after importing package)

# generate sample from tridiagonal precision matrix
p = 1000
n = 10
l = .6

test="trid"

np.random.seed(1)

# generate a tridiagonal matrix

if(test=="trid"):
	print("test=trid")
	a = -0.5 * np.ones(p-1)
	b = 1.25 * np.ones(p)
	iC_star = np.diag(a,-1) + np.diag(b,0) + np.diag(a,1) 

if(test=="rand"):
	print("test=rand")
	iC_star = np.random.randn(p,p)
	iC_star = (iC_star+iC_star.transpose())/2

	iC_star = iC_star * (abs(iC_star)>.5 * np.max(abs(iC_star)))

	for i in range(0,p):
		iC_star[i,i] =0.0
		iC_star[i,i] = np.sum(abs(iC_star[:,i]))+.1

# generate the data
L = np.linalg.cholesky(iC_star)
Y = np.linalg.solve(L.T,np.random.randn(p,n))


nnzpr=np.count_nonzero(iC_star)/p


print(">>>>>>>>nnzpr = ",nnzpr)


sd = np.sqrt(Y.var(1))

for i in range(0,p):
	Y[i,:]=Y[i,:]/sd[i];



#SQUIC.PATH_TO_libSQUIC('/Users/aryan')
SQUIC.PATH_TO_libSQUIC('/Users/aryan/Documents/code/libSQUIC/build')
[X_new,W_new,info_times_new,info_objective_new,info_logdetX_new,info_trSX_new] = SQUIC.run(Y,l,max_iter=100,tol=1e-3,verbose=1)


#[X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC.run(Y,l,max_iter=100,tol=1e-12,verbose=1)


#print("out_obj=",np.round(info_objective_new,3),";")

#print(np.round(W_new.todense(),3))


#print(np.linalg.det(W_new.todense()))

#





#print(np.round(X_new.todense(),3))


#print(np.round(X.todense(),3))



#for i in range(0,p):
#	Y[i,:]=Y[i,:]/sd[i];



#print(Y)



#check
#np.cov(Y,bias=1)



#print(Y)
#print(np.mean(Y,1))



#X_dense = np.array(X.todense())



#Estimate_bool = (abs(X_dense)>1e-12)
#Estimate_bool = Estimate_bool.flatten()
#Estimate_bool = Estimate_bool[0]



#Truth_bool = (abs(iC_star)>1e-12)
#Truth_bool = Truth_bool.flatten()
#Truth_bool = Truth_bool[0]





#f1 = f1_score(Truth_bool, Estimate_bool)


#print(f1)


#[X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC.run(Y,l,max_iter=100,tol=1e-3)


#print(X.todense().round(2))

#T=X.todense().round(2)

#T=T[0:20,0:20]
#plt.figure
#plt.imshow(T, cmap='hot')
#plt.show()


#plt.figure
#plt.spy(X[0:20,0:20])
#plt.show()


