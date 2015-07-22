using DataFrames

A = readdlm("u1.base",'\t';header=false)
file = "u1.base"

function pearsonCorrelation(x,y)
        @assert(length(x) == length(y))
        n = length(x)
        @assert(n>0)
        avg_x = mean(x)
        avg_y = mean(y)
        prod_diff = 0
        xdiff2 = 0
        ydiff2 = 0
        for id = 1:n
                xdiff = x[id] - avg_x
                ydiff = y[id] - avg_y
                prod_diff += xdiff*ydiff
                xdiff2 += xdiff^2
                ydiff2 += ydiff^2
        end
        return prod_diff/sqrt(xdiff2*ydiff2)
end

function PearsonCorrelationHelper(file)
     df = readtable(file,separator = '\t',header = false);
     rename!(df,:x1,:userId)
     rename!(df,:x2,:itemId)
     rename!(df,:x3,:rating)
     #rename!(df,:x4,:timestamp)
     M1 =Set(df[df[:userId] .== 1, :itemId])
     CorrelationArray = Float64[] 
     (N,ind) = findmax(df[:,1])
     for(i = 2:N)
            M_i = Set(df[df[:userId] .== i, :itemId])
            if(length(intersect(M1,M_i)) == 0) push!( CorrelationArray , NA)
            else
                  X = Float64[]
                  Y = Float64[]
                  for(movie in intersect(M1,M_i))
                      push!(X , (df[(df[:userId]   .== 1)  & (df[:itemId] .== movie), :rating])[1])
                      push!(Y , (df[(df[:userId]   .== i)  & (df[:itemId] .== movie), :rating])[1])
                  end
                  push!( CorrelationArray , pearsonCorrelation(X,Y))
            end
     end
end
