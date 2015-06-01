#A = readdlm("/home/abhijith/Documents/Viral/codes/ml-100k/u1.base.txt",'\t';header=false)
A = readdlm("u1.base",'\t';header=false)
I = readdlm("movies.csv",',';header=false)
#Q = readdlm("/home/abhijith/Documents/Viral/codes/ml-100k/u1.test.txt",'\t';header=false)

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
N_f = 3
MM = rand(n_m,N_f-1)
FirstRow=zeros(Float64,n_m)
for i=1:n_m
    FirstRow[i]=mean(full(nonzeros(R[:,1])))
end
M = [FirstRow';MM']
(r,c,v)=findnz(R)
II=sparse(r,c,1)
locWtU=sum(II,2)
locWtM=sum(II,1)
LamI=lambda*eye(N_f)
U=zeros(n_u,N_f)
noIters=30

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

function recommend(user,n)
    # All the movies sorted in decreasing order of rating.
    top = sortperm(vec(U[user,:]*M))
    # Movies seen by user
    m = find(R[user,:])    
    unseen_top = setdiff(Set(top),Set(m))
    movie_names = readdlm("movies.csv",'\,')
    println(movie_names[[collect(unseen_top)[1:n,:][:]],2])
end
