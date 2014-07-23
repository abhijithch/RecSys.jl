function PrintFactorNames(matrix,rowID::Int64,num::Int64)
    matrixwithindex = AddIndices(matrix)
    sortedn = sortcols(matrixwithindex,by=x->(x[rowID+1]))
    topn = sortedn[1,1:num]
    bottomn = sortedn[1,m-num+1:m]
    PrintNames(topn,bottomn,rowID)
    #return topn,bottomn
end

function ItemName(ItemID::Int64)
    return ItemsDict[ItemID]
end

function AddIndices(matrix)
    matrixwithindex = [[1:8]',M]
    return matrixwithindex
end

function PrintNames(top,bottom,factorID)
    println("Factor ",factorID)
    for i=1:length(top)
        @printf "%4.3f " a[factorID+1, int(top[i])]
        println(ItemName(int(top[i])))
    end
    println("...")
    for i=1:length(bottom)
        @printf "%4.3f " a[factorID+1, int(bottom[i])]
        println(ItemName(int(bottom[i])))
    end
    println()
end
