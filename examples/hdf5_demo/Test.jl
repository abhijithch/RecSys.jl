using HDF5
cd("/RecSys.jl/examples/hdf5_demo/")
file = h5open("test.h5", "w")
g = g_create(file, "mygroup") # create a group
g["dset1"] = 3.2              # create a scalar dataset
attrs(g)["Description"] = "This group contains only a single dataset" # an attribute
close(file)
