using DataFrames
using MLBase

# try to follow Julia style guide at:
# http://julia.readthedocs.org/en/latest/manual/style-guide/

# function takes in a string as format and calls appropriate indermediate function
#
# all read functions return a DataFrame with a common first three columns and
# extra info appended in other columns
function readdata(file, format)

end

function read_movielens(file)

end

function read_netflix(file)
  # datset format at:
  # https://gist.github.com/janisozaur/3192952

end

function read_jester(file)
  # http://www.ieor.berkeley.edu/%7Egoldberg/jester-data/

end

function read_lastfm(file)
  # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html

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
