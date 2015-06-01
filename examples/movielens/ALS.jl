#A = readdlm("/home/abhijith/Documents/Viral/codes/ml-100k/u1.base.txt",'\t';has_header=false)
A = readdlm("u1.base",'\t';has_header=false)
I = readdlm("movies.csv",',';has_header=false)
#Q = readdlm("/home/abhijith/Documents/Viral/codes/ml-100k/u1.test.txt",'\t';has_header=false)

userCol = int(A[:,1])
movieCol = int(A[:,2])
ratingsCol = int(A[:,3])
tempR=sparse(userCol,movieCol,ratingsCol)

(n_u,n_m)=size(tempR)
tempR_t=tempR'

#Filter out empty movies or users.
indd_users=trues(n_u)
for u=1:n_u
    movies=find(tempR_t[:,u])
    if length(movies)==0
       indd_users[u]=false
    end
end
tempR=tempR[indd_users,:]
indd_movies=trues(n_m)
for m=1:n_m
    users=find(tempR[:,m])
    if length(users)==0
       indd_movies[m]=false
    end
end
tempR=tempR[:,indd_movies]
R=tempR
R_t=R'
(n_u,n_m)=size(R)
lambda = 0.065
N_f = 20
MM = rand(n_m,N_f-1)
FirstRow=zeros(Float64,n_m)
for i=1:n_m
    FirstRow[i]=mean(nonzeros(full(R[:,1])))
end
M = [FirstRow';MM']
(r,c,v)=findnz(R)
II=sparse(r,c,1)
locWtU=sum(II,2)
locWtM=sum(II,1)
LamI=lambda*eye(N_f)
U=zeros(n_u,N_f)
noIters=10

for i=1:noIters
    for u=1:n_u
    	#println(u)
        movies=find(R_t[:,u])
        M_u=M[:,movies]
        vector=M_u*full(R_t[movies,u])
        matrix=(M_u*M_u')+locWtU[u]*LamI
        x=matrix\vector
        U[u,:]=x
        #println(round(x,2))
    end
  #println(i)
    for m=1:n_m
    	#println(m)
  	users=find(R[:,m])
        U_m=U[users,:]
        vector=U_m'*full(R[users,m])
        matrix=(U_m'*U_m)+locWtM[m]*LamI
        x=matrix\vector
	#println("OK")
        M[:,m]=x
     end

end

#println(round(U*10))
#println(round(M*10))

 (m,n) = size(M)

 # Adding Indices
 a = [[1:n]',M]


 sorted1 = sortcols(a,by=x->(x[2]))
 sorted2 = sortcols(a,by=x->(x[3]))

 ItemsDict = Dict()

 (m,n) = size(I)

 for i=1:m
    setindex!(ItemsDict,I[i,2],int(I[i,1]))
 end

function AddIndices(matrix)
    matrixwithindex = [[1:8]',M]
    return matrixwithindex
end

function PrintFactorNames(matrix,rowID::Int64,num::Int64)
    matrixwithindex = AddIndices(matrix)
    sortedn = sortcols(matrixwithindex,by=x->(x[rowID+1]))
    topn = sortedn[1,1:num]
    bottomn = sortedn[1,m-num+1:m]
    PrintNames(topn,bottomn,rowID)
    #return topn,bottomn
end

function ItemName(ItemID::Int64)
    return ItemsDict[ItemID]
end

function PrintNames(top,bottom,factorID)
    println("Factor ",factorID)
    for i=1:length(top)
        @printf "%4.3f " a[factorID+1, int(top[i])]
        println(ItemName(int(top[i])))
    end
    println("...")
    for i=1:length(bottom)
        @printf "%4.3f " a[factorID+1, int(bottom[i])]
        println(ItemName(int(bottom[i])))
    end
    println()
end



#    PrintFactorNames(M,1,3)

#    PrintFactorNames(M,2,3)

#    PrintFactorNames(M,3,3)
