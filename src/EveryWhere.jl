
@everywhere function gatherItemMatrix(remoteRefOfItemMatrix::Array, noOfWorkers)
	#Gather ItemMatrix from all processors to local processors
	fullItemMatrix = Array{Float64,2}
    for k in 1:noOfWorkers
    	if k==1			    		    		
    		fullItemMatrix =  fetch(remoteRefOfItemMatrix[k])'			    		
    	else
    		fullItemMatrix = [fullItemMatrix;(fetch(remoteRefOfItemMatrix[k]))']
    	end
    end	    
    return fullItemMatrix
end	

@everywhere function gatherUserMatrix(remoteRefOfUserMatrix::Array, noOfWorkers)
	uMatrix = Array{Float64,2}
    for k in 1:noOfWorkers
    	if k == 1
    		uMatrix = fetch(remoteRefOfUserMatrix[k])
    	else
    		uMatrix = [uMatrix;fetch(remoteRefOfUserMatrix[k])]
    	end
    end
    return uMatrix
end

@everywhere function findU(remoteRefOfTraningDataByRow::RemoteRef, remoteRefOfItemMatrix::Array, noOfWorkers::Int32, noOfUsers)												   

	fullItemMatrix = gatherItemMatrix(remoteRefOfItemMatrix, noOfWorkers)
    fullItemMatrix = fullItemMatrix'			    			   
    #println("fullItemMatrix",fullItemMatrix)
	ratingMatrix = fetch(remoteRefOfTraningDataByRow)
    #println("ratingMatrix",ratingMatrix)
    #println("size(ratingMatrix)",size(ratingMatrix))
	xMatrix = Array(Float64, size(ratingMatrix)[1], noOfUsers)
	println("xMatrix size",size(xMatrix))
	for r = 1:size(ratingMatrix)[1]		
		ratingMatrixTranspose = (ratingMatrix[r,:])'									
    	items=find(ratingMatrixTranspose)  			    			   			    				    	
        #println("ratingMatrixTranspose",ratingMatrixTranspose)
        #println("items",items)
        #println("fullItemMatrix[:,items]",fullItemMatrix[:,items])
    	vector = (fullItemMatrix[:,items] ) * full(ratingMatrixTranspose)[items]
        #println("size of vector",size(vector))
    	matrix=(fullItemMatrix[:,items]*(fullItemMatrix[:,items])')			    				    	    	
        #println("size of matrix",size(matrix))
        #println("size of xMatrix",size(xMatrix))
        #println("matrix-vector   ",matrix\vector)
        #println("size matrix-vector  ",size(matrix\vector ))
    	xMatrix[r,:]=matrix\vector           			    
    end    
    xMatrix
end	

@everywhere function findM(remoteRefOfTraningDataByColumn::RemoteRef, remoteRefOfUserMatrix::Array, noOfWorkers::Int32, noOfUsers)				
	
	uMatrix = gatherUserMatrix(remoteRefOfUserMatrix, noOfWorkers)

	ratingMatrix = fetch(remoteRefOfTraningDataByColumn)				
	xMatrix = Array(Float64, noOfUsers, size(ratingMatrix)[2])
	
	for c = 1:size(ratingMatrix)[2]
		ratingMatrixCol = ratingMatrix[:,c]
		users=find(ratingMatrixCol)								   					
	    vector = (uMatrix[users,:] )' * full(ratingMatrixCol)[users]				    				    
	    matrix = ((uMatrix[users,:])'*(uMatrix[users,:]))		    	
	    xMatrix[:,c]=(matrix\vector)                				    
	end	
	xMatrix
end				
