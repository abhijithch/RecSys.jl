
function filterNonParticipatingUsersAndItems(trainingData::SparseMatrixCSC)	
	trainingDataTranspose = trainingData'

	(noOfUsers, noOfItems) = size(trainingData)

	#Filter the users who have not rated. any item.
	ratedUsers=trues(noOfUsers)
	for u = 1:noOfUsers
	    itemsRatedByUser = find(trainingDataTranspose[:,u])
	    if length(itemsRatedByUser) == 0
	       ratedUsers[u] = false
	    end
	end
	trainingData=trainingData[ratedUsers,:]

	#Filter the items that are not rated by any user.
	ratedItems=trues(noOfItems)
	for i =1:noOfItems
	    userRatedItems = find(trainingData[:,i])
	    if length(userRatedItems) == 0
	       ratedItems[i]=false
	    end
	end
	trainingData=trainingData[:,ratedItems]
end

function initializeItemMatrix(trainingData, noOfItems, numberOfFeatures)
	
	# itemMatrix = rand(noOfItems,numberOfFeatures-1)	
	# FirstRow=zeros(Float64,noOfItems)

	# for i = 1:noOfItems
	#     FirstRow[i]=mean(nonzeros(trainingData[:,i]))
	# end

	# finalItemMatrix = [FirstRow';itemMatrix']
	# println(finalItemMatrix)
	# return finalItemMatrix

	(U,S,V) = svd(full(trainingData))
	M = diagm(S) * V'
	return M

end

function distributeMatrixByColumn(matrix, noOfWorkers, totalSize)
	remoteRefOfMatrix = Array(RemoteRef,noOfWorkers)

	minColumnPerProc = floor(totalSize/noOfWorkers)
	additionalColumnForLastProc = totalSize%noOfWorkers

	for w  in 1:noOfWorkers	    
	    remoteRefOfMatrix[w] =  RemoteRef(w+1)
	    if w == noOfWorkers
	    	put!(remoteRefOfMatrix[w],matrix[:,[minColumnPerProc*(w-1)+1:end]])
	    else
	    	put!(remoteRefOfMatrix[w],matrix[:,[minColumnPerProc*(w-1)+1:minColumnPerProc*w]])
	    end
	end
	return remoteRefOfMatrix
end

function distributeMatrixByRow(matrix, noOfWorkers, totalSize)
	remoteRefOfMatrix = Array(RemoteRef,noOfWorkers)

	minRowPerProc = floor(totalSize/noOfWorkers)
	additionalRowForLastProc = totalSize%noOfWorkers

	for w  in 1:noOfWorkers	    
	    remoteRefOfMatrix[w] =  RemoteRef(w+1)
	    if w == noOfWorkers
	    	put!(remoteRefOfMatrix[w],matrix[[minRowPerProc*(w-1)+1:end],:])
	    else
	    	put!(remoteRefOfMatrix[w],matrix[[minRowPerProc*(w-1)+1:minRowPerProc*w],:])
	    end
	end
	return remoteRefOfMatrix
end
