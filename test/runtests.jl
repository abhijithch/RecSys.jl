using FactCheck
using JuliaRecSys


facts("Testing filterNonParticipatingUsersAndItems") do
       
		RatingMarix = [ 3 4 5 0; 0 0 0 0; 0 6 7 0]
		sparseRatingMatrix = sparse(RatingMatrix)

       context("Remove a row and column") do
         @fact full(filterNonParticipatingUsersAndItems(sparseRatingMatrix)) => [ 3 4 5; 0 6 7]
       end
           
end

# a=loadData("",'\t')
# a=loadData("/home/thiruk/JuliaREC/ALS_PG/JuliaRecSys.jl/test/testdata/1.txt",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')
# a=loadData("",'\t')