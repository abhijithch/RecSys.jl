function ALSFactorization(trainingData::SparseMatrixCSC, numberOfFeatures, noOfIterations)
		
    trainingData = filterNonParticipatingUsersAndItems(trainingData)
    #println(trainingData)
    trainingDataTranspose = trainingData'
    (noOfUsers, noOfItems) = size(trainingData)		
    if nprocs() == 1
        noOfWorkers = 1
    else
        noOfWorkers = nprocs()-1
    end
	# if noOfWorkers == 0
	# 	return
	# end

	# if noOfUsers < noOfWorkers | noOfItems < noOfWorkers
	# 	return
	# end

	println("No of items",noOfItems)
	println("No of workers",noOfWorkers)
	println("No of users",noOfUsers)
	itemMatrix = initializeItemMatrix(trainingData, noOfItems, numberOfFeatures)
	#println(itemMatrix)
	userMatrix = zeros(noOfUsers, numberOfFeatures)
	#println(userMatrix)
    remoteRefOfItemMatrix = distributeMatrixByColumn(itemMatrix, noOfWorkers, noOfItems)	
    remoteRefOfUserMatrix = distributeMatrixByRow(userMatrix, noOfWorkers, noOfUsers)
    remoteRefOfTraningDataByRow = distributeMatrixByRow(trainingData, noOfWorkers, noOfUsers)
    remoteRefOfTraningDataByColumn = distributeMatrixByColumn(trainingData, noOfWorkers, noOfItems)
    
    for iter = 1: noOfIterations
        println(iter)
	@sync begin
	    for (widx, worker) in enumerate(workers())										
		remoteRefOfUserMatrix[widx] = @spawnat worker findU(remoteRefOfTraningDataByRow[widx], remoteRefOfItemMatrix, noOfWorkers, noOfUsers)							
	    end
        end
	@sync begin
            for (widx, worker) in enumerate(workers())	
		remoteRefOfItemMatrix[widx] = @spawnat worker findM(remoteRefOfTraningDataByColumn[widx], remoteRefOfUserMatrix, noOfWorkers, noOfUsers)							
	    end
	end       
    end
    
	#reconstrut the U and M
	ItemMatrix = gatherItemMatrix(remoteRefOfItemMatrix, noOfWorkers)
	UserMatrix = gatherUserMatrix(remoteRefOfUserMatrix, noOfWorkers)
	#UserMatrix * ItemMatrix'
	#Lumberjack.info(logLM,"loadData() method","here")
#	return (UserMatrix, ItemMatrix)
	return ItemMatrix

end
