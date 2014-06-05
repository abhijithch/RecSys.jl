
function findRootMeanSquareError(UserMatrix, ItemMatrix, testratings,stepRange)
    
    (rownum , columnnum , values)=findnz(testratings)
    noOfValues=length(values)
    predicted = zeros(noOfValues)    
    i =  int(floor((stepRange.stop-stepRange.start)/stepRange.step)+1)
    rmseArray=zeros(i)
    k = 1
    for index=stepRange
        rowfeature=UserMatrix[:,1:index]        
        colfeature=ItemMatrix[1:index,:]
                
        for j=1:noOfValues           
           rowfeaturetemp=rowfeature[rownum[j],:]
           P=rowfeaturetemp[:]
           Q=colfeature[:,columnnum[j]]                    
           predicted[j]=dot(P,Q)	
        end

        predicted[find(predicted.<1)]=1        
        predicted[find(predicted.>5)]=5            
        rmseArray[k] = sqrt(sum((float(testratings[:]) - predicted[:]).^2) / noOfValues )            
        k=k+1
    end
    
    rmseArray
end