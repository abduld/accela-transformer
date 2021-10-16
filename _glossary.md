
#### maxComputeWorkGroupSize
Is the maximum size of a local compute workgroup, per dimension. These three values represent the maximum local workgroup size in the X, Y, and Z dimensions, respectively. The x, y, and z sizes, as specified by the LocalSize execution mode or by the object decorated by the WorkgroupSize decoration in shader modules, must be less than or equal to the corresponding limit.
This is the maximum thread counts in x/y/z, and you'll find it's most often 1024x1024x64 on Nvidia, and 256x256x256 on AMD and Intel. I don't know about mobile GPUs.

