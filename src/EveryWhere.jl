
function gatherItemMatrix(remoteRefOfItemMatrix::Array, noOfWorkers)	
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

 function gatherUserMatrix(remoteRefOfUserMatrix::Array, noOfWorkers)
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

 function findU(remoteRefOfTraningDataByRow::RemoteRef, remoteRefOfItemMatrix::Array, noOfWorkers::Int32, noOfUsers)												   

	fullItemMatrix = gatherItemMatrix(remoteRefOfItemMatrix, noOfWorkers)
    fullItemMatrix = fullItemMatrix'			    			       
	ratingMatrix = fetch(remoteRefOfTraningDataByRow)    
	xMatrix = Array(Float64, size(ratingMatrix)[1], noOfUsers)	

	for r = 1:size(ratingMatrix)[1]		
		ratingMatrixTranspose = (ratingMatrix[r,:])'									
    	items=find(ratingMatrixTranspose)  			    			   			    				    	        
    	vector = (fullItemMatrix[:,items] ) * full(ratingMatrixTranspose)[items]        
    	matrix=(fullItemMatrix[:,items]*(fullItemMatrix[:,items])')			    				    	    	        
    	xMatrix[r,:]=matrix\vector           			    
    end    
    xMatrix
end	

 function findM(remoteRefOfTraningDataByColumn::RemoteRef, remoteRefOfUserMatrix::Array, noOfWorkers::Int32, noOfUsers)				
	
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
