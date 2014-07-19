using Lumberjack 


logLM = Lumberjack.LumberMill()
add_truck(logLM, LumberjackTruck("../alsfactorization.log","info"), "my-file-logger")
function loadData(fileLocation::String, fileSeparator::Char)
	try
		#Lumberjack.info(logLM,"loadData() method",{"fileLocation"=>fileLocation ,"fileSeparator"=>fileSeparator})
		if !isfile(fileLocation)			
			Lumberjack.warn(logLM,"loadData() method",{"message"=>"Please provide a valid file location"})
			return
		end
		#Lumberjack.info(logLM,"loadData() method",{"file size"=>filesize(fileLocation)})		
		data = readdlm(fileLocation, fileSeparator; has_header=false)	
		#Lumberjack.info(logLM,"loadData() method",{"data"=>data})	
		userColumn = int(data[:,1])
		itemColumn = int(data[:,2])
		ratingsColumn = int(data[:,3])
		dataSparse = sparse(userColumn,itemColumn,ratingsColumn)
	catch exp
		Lumberjack.error(logLM, "error loading data", { "errormsg"=>exp,"fileLocation"=>fileLocation,"fileSeparator"=>fileSeparator})
	end
end

