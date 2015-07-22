using DataFrames

A = readdlm("u1.base",'\t';header=false)
file = "u1.base"

function SpearmanCorrelation(x,y)
  @assert(length(x) == length(y))
  n = length(x)
  @assert(n>0)
	RankX    = Int64[]
	RankY    = Int64[]
  d        = Float64[]
  dsquared = Float64[]
	IndexX   = sortperm(x)
	IndexY   = sortperm(y)
  SumD2    = Float64{}
  SumD2    = 0.0
	for(i = 1: length(x))
		push!(RankX,0)
		push!(RankY,0)
    push!(d,0.0)
    push!(dsquared,0.0)
	end
	for(i = 1:length(x))
		RankX[IndexX[i]] = RankY[IndexY[i]] = length(x) - i + 1	
	end
  for(i = 1:length(x))
    d[i]          = RankX[i] - RankY[i]
    dsquared[i]   = d * d
    SumD2        += dsquared[i] 
  end
  spearCoeff = 1 - 6 * SumD2/(N^3 - N)
  return spearCoeff
end

function SpearmanCorrelationHelper(file)
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
                  push!( CorrelationArray , SpearmanCorrelation(X,Y))
            end
     end
end
