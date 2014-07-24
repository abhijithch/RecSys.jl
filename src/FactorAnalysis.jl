function PrintFactorNames(matrix,itemsdict,rowID::Int64,num::Int64)
    matrixwithindex = AddIndices(matrix)
    sortedn = sortcols(matrixwithindex,by=x->(x[rowID+1]))
    topn = sortedn[1,1:num]
    bottomn = sortedn[1,m-num+1:m]
    PrintNames(matrixwithindex,itemsdict,topn,bottomn,rowID)
    #return topn,bottomn
end

function PrintFactorNamesWithIndices(matrix,itemsdict,rowID::Int64,num::Int64)
    matrixwithindex = matrix
    (m,n) = size(matrix)
    sortedn = sortcols(matrixwithindex,by=x->(x[rowID+1]))
    topn = sortedn[1,1:num]
    bottomn = sortedn[1,n-num+1:n]
    PrintNames(matrixwithindex,itemsdict,topn,bottomn,rowID)
    #return topn,bottomn
end

function ItemName(itemsdict,ItemID::Int64)
    return itemsdict[ItemID]
end

function ReadItems(Filepath)
    items = readdlm(Filepath,',';has_header=false)
    itemsdict = Dict()
    (m,n) = size(items)
    for i=1:m
        setindex!(itemsdict,items[i,2],int(items[i,1]))
    end
    return itemsdict
end

function AddIndices(matrix)
    (m,n) = size(matrix)
    matrixwithindex = [[1:n]',matrix]
    return matrixwithindex
end

function PrintNames(matrix,itemsdict,top,bottom,factorID)
    println("Factor ",factorID)
    for i=1:length(top)
        @printf "%4.3f " matrix[factorID+1, int(top[i])]
        println(itemsdict[int(top[i])])
    end
    println("...")
    for i=1:length(bottom)
        @printf "%4.3f " matrix[factorID+1, int(bottom[i])]
        println(itemsdict[int(bottom[i])])
    end
    println()
end
