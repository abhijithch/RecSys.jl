using DataFrames
using MLBase
using Taro
Taro.init()

# try to follow Julia style guide at:
# http://julia.readthedocs.org/en/latest/manual/style-guide/

# function takes in a string as format and calls appropriate indermediate function
#
# all read functions return a DataFrame with a common first three columns and
# extra info appended in other columns
function readdata(file, format)

end

function read_movielens(file)
  
#The movielens data format is as follows :
#Users and items are numbered consecutively from 1.
#The data is randomly ordered. This is a tab separated list of
#user id | item id | rating | timestamp.

  df = readtable(file,separator = '\t',header = false);
  rename!(df,:x1,:userId)
  rename!(df,:x2,:itemId)
  rename!(df,:x3,:rating)
  rename!(df,:x4,:timestamp)
  matrix(df::DataMatrix, na = true)
  return df

end

function read_netflix(file)
  # datset format at:
  # https://gist.github.com/janisozaur/3192952
  ### Ratings are withheld for the netflix data
  ### The DataFrame is an array of userIds,itemIds and timestamps only!!!
  f  = open(file);
  df = DataFrame(userId,itemId,timestamp)
  for line in eachline(f):
	  if(line[end-1] == ':')
	      movieId = line[:end-1]
	  else
              parts = split(ln,",")
              push!(df,[movieId,parts[0],parts[1]])
          end
end

function read_jester(file)
  # http://www.ieor.berkeley.edu/%7Egoldberg/jester-data/
  # /home/pramod/Desktop/RecSysData/Various DataSets/jester-data-1.xls
  f  = open(file);
  dfJesterTemp = Taro.readxl("/home/pramod/Desktop/RecSysData/Various_DataSets/jester-data-1.xls",
                         "jester-data-1-new", 
			 "A1:CW24983"; 
			  header=false)
  delete!(dfJesterTemp, :x2)			  
  [dfJesterTemp[dfJesterTemp[nm] .== 99, nm] = -100 for nm in names(dfJesterTemp)]
  (rows,cols) = size(dfJesterTemp)
  JesterData  = DataFrame(userId = Int64[], itemID = Int64[] , rating = Float64[])
  for r = 1:rows
    for c = 1:cols
         push!(JesterData,[r,c,dfJesterTemp[r,c]])
    end
  end

end

function read_lastfm(file)
  # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html
  df = readtable(file,separator = '\t',header = false);
  rename!(df,:x1,:userId)
  rename!(df,:x2,:artistId)
  rename!(df,:x3,:rating)
  matrix(df::DataMatrix, na = true)
  return df
end

function read_yahoomusic(file)
  # http://webscope.sandbox.yahoo.com/catalog.php?datatype=r

end

function read_mahout(file)
  # needs more discussion/ thought

end

function read_eachmovie(file)
  # http://www.cs.cmu.edu/~lebanon/IR-lab/data.html#obtaining
  # may ignore for now

end

# returns sparse matrix representation given the data of U and M matrices
function getmatrix(data::DataFrame)

  # rough code from demo/ALS
  # rewrite from scratch

  # userCol = int(A[:,1])
  # movieCol = int(A[:,2])
  # ratingsCol = int(A[:,3])
  # tempR=sparse(userCol,movieCol,ratingsCol)
  #
  # (n_u,n_m)=size(tempR)
  # tempR_t=tempR'
  #
  # #Filter out empty movies or users.
  # indd_users=trues(n_u)
  # for u=1:n_u
  #     movies=find(tempR_t[:,u])
  #     if length(movies)==0
  #        indd_users[u]=false
  #     end
  # end
  # tempR=tempR[indd_users,:]
  # indd_movies=trues(n_m)
  # for m=1:n_m
  #     users=find(tempR[:,m])
  #     if length(users)==0
  #        indd_movies[m]=false
  #     end
  # end

  # returning multiple matrices
  # http://stackoverflow.com/questions/27095173/how-can-a-function-have-multiple-return-values-in-julia-vs-matlab
end

function eval(result)
  # evalulate results with MLBase
  # needs more thought, can come later
end
