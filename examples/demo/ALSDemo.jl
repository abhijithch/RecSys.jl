function Rating(path::ASCIIString,delim,h)
    #A = readdlm("../../ml-100k/u1.base",'\t';header=false)
    #T = readdlm("../../ml-100k/u1.test",'\t';header=false)
    A = readdlm(path,delim;header=h)
    #The format is userId , movieId , rating
    if !h
        userCol = int(A[:,1])
        movieCol = int(A[:,2])
        ratingsCol = float(A[:,3])
    else
	userCol = int(A[1][:,1])
        movieCol = int(A[1][:,2])
        ratingsCol = float(A[1][:,3])
    end
    #Create Sparse Matrix
    tempR=sparse(userCol,movieCol,ratingsCol)
    #println(tempR)
    return tempR
end



function Prepare(R::SparseMatrixCSC{Float64,Int64}, noFactors::Int64)
    N_f = noFactors
    R_t=R'
    #Filter out empty movies or users.
    users=sum(R_t,1)
    indd_users=find(users)
    R=R[indd_users,:]
    movies=sum(R,1)
    indd_movies=find(movies)
    R=R[:,indd_movies]
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
    return U, M, R
end

function factorize(Ra::SparseMatrixCSC{Float64,Int64},noIters::Int64,noFactors::Int64)
    U, M, R = Prepare(Ra,noFactors)
    (n_u,k)=size(U)
    (k,n_m)=size(M)
    noIters=noIters
    R_t = R'
    N_f = noFactors
    lambda = 0.065
    (r,c,v)=findnz(R)
    II=sparse(r,c,1)
    locWtU=sum(II,2)
    locWtM=sum(II,1)
    LamI=lambda*eye(N_f)
    movies=Dict{Int64,Any}()
    for u=1:n_u
        movies[u]=find(R_t[:,u])
    end
    users=Dict{Int64,Any}()
    for m=1:n_m
        users[m]=find(R[:,m])
    end
    #The Alternate Least Squares(ALS)
    for i=1:noIters
        for u=1:n_u
    	    #Update U
            M_u=M[:,movies[u]]
            #vector=Array{Float64,2}
            vector=M_u*full(R_t[movies[u],u])
            #matrix=Array{Float64,2}
            matrix=(M_u*M_u')+locWtU[u]*LamI
            x=matrix\vector
            U[u,:]=x
            #println(round(x,2))
        end
        #println(i)
        for m=1:n_m
    	    #Update M
            U_m=U[users[m],:]
            #vector=Array{Float64,2}
            vector=U_m'*full(R[users[m],m])
            #matrix=Array{Float64,2}
            matrix=(U_m'*U_m)+locWtM[m]*LamI
            x=matrix\vector
	    #println("OK")
            M[:,m]=x
        end
    end
    return U, M
end

# Function to save the computed
function save(U,M,R,ftype)

end

function load(ftype)

end

         
function update_R()

end

function top_factors()

end



function recommend(U, M, R,user,n)
    # All the movies sorted in decreasing order of rating.
    top = sortperm(vec(U[user,:]*M))
    # Movies seen by user
    m = find(R[user,:])    
    # unseen_top = setdiff(Set(top),Set(m))
    # To Do: remove the intersection of seen movies.  
    movie_names = readdlm("./data/movies.csv",'\,')
    movie_names[top[1:n,:][:],2]
end

