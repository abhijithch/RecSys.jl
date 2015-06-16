using DataFrames

#For example, file : "/home/pramod/Desktop/RecSysData/u.data"
#Users and items are numbered consecutively from 1.
#The data is randomly ordered. This is a tab separated list of
#user id | item id | rating | timestamp.

function ReadDataRecSys(file)
  df = readtable(file,separator = '\t',header = false);
  rename!(df,:x1,:userId)
  rename!(df,:x2,:itemId)
  rename!(df,:x3,:rating)
  rename!(df,:x4,:timestamp)
  return df
end
