function Rating(path::ASCIIString,delim)
    #A = readdlm("../../ml-100k/u1.base",'\t';header=false)
    #T = readdlm("../../ml-100k/u1.test",'\t';header=false)
    A = readdlm(path,delim,header=false)
    #The format is userId , movieId , rating
    userCol = int(A[:,1])
    movieCol = int(A[:,2])
    ratingsCol = int(A[:,3])
    #Create Sparse Matrix
    tempR=sparse(userCol,movieCol,ratingsCol)
    #println(tempR)
    return tempR
end

function Prepare(tempR::SparseMatrixCSC{Int64,Int64}, noFactors::Int64)
    N_f = noFactors
    (n_u,n_m)=size(tempR)
    println(n_u)
    println(n_m)
    tempR_t=tempR'
    #Filter out empty movies or users.
    indd_users=trues(n_u)
    println(size(indd_users))
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
    #Using Parameters lambda and N_f
    #lambda related to regularization and cross validation
    #N_f is the dimension of the feature space
    MM = rand(n_m,N_f-1)
    FirstRow=zeros(Float64,n_m)
    for i=1:n_m
        FirstRow[i]=mean(full(nonzeros(R[:,i])))
    end
    #Update FirstRow as mean of nonZeros of R 
    M = [FirstRow';MM']
    U=zeros(n_u,N_f)
    return U, M
end

function factorize(R::SparseMatrixCSC{Int64,Int64},noIters::Int64,noFactors::Int64)
    (n_u,n_m)=size(R)
    N_f = noFactors
    lambda = 0.065
    (r,c,v)=findnz(R)
    II=sparse(r,c,1)
    locWtU=sum(II,2)
    locWtM=sum(II,1)
    LamI=lambda*eye(N_f)
    U, M = Prepare(R,noFactors)
    noIters=noIters
    R_t = R'
    #The Alternate Least Squares(ALS)
    for i=1:noIters
        for u=1:n_u
    	    #Update U
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
    	    #Update M
  	    users=find(R[:,m])
            U_m=U[users,:]
            vector=U_m'*full(R[users,m])
            matrix=(U_m'*U_m)+locWtM[m]*LamI
            x=matrix\vector
	    #println("OK")
            M[:,m]=x
        end
    end
    return U, M
end

#=
function recommend(U, M, R,user,n)
    # All the movies sorted in decreasing order of rating.
    top = sortperm(vec(U[user,:]*M))
    # Movies seen by user
    m = find(R[user,:])    
    # unseen_top = setdiff(Set(top),Set(m))
    # To Do: remove the intersection of seen movies.  
    movie_names = readdlm("movies.csv",'\,')
    movie_names[top[1:n,:][:],2]
end
=#
