
function loadData(fileLocation::String, fileSeparator::Char)
	try		
		if !isfile(fileLocation)			
			println("Please provide a valid file location")
			return
		end
		
		data = readdlm(fileLocation, fileSeparator; has_header=false)	
		
		userColumn = int(data[:,1])
		itemColumn = int(data[:,2])
		ratingsColumn = int(data[:,3])
		dataSparse = sparse(userColumn,itemColumn,ratingsColumn)
	catch exp
		println("Error loading data")
	end
end

