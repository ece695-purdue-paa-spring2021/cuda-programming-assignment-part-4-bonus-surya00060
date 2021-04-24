
#include "cudaLib.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}


//	STUDENT: Add functions here
// Part 4 Starts here
TensorShape ComputeConvOutput(TensorShape iShape, TensorShape fShape, ConvLayerArgs args)
{
	TensorShape oShape;
	oShape.height 	= (iShape.height + 2 * args.padH - fShape.height) / args.strideH + 1;
	oShape.width	= (iShape.width  + 2 * args.padW - fShape.width)  / args.strideW + 1;
	oShape.channels	= (fShape.count);
	oShape.count 	= iShape.count;
	// std::cout<<oShape<<std::endl;
	return oShape;
}

TensorShape ComputePoolOutput(TensorShape iShape, PoolLayerArgs args)
{
	TensorShape oShape;
	oShape.height 	= (iShape.height - args.poolH) / args.strideH + 1;
	oShape.width	= (iShape.width  - args.poolW) / args.strideW + 1;
	oShape.channels	= iShape.channels;
	oShape.count 	= iShape.count;
	// std::cout<<oShape<<std::endl;
	return oShape;
}

TensorShape ComputeFCOutput(TensorShape aShape, TensorShape bShape)
{
	TensorShape cShape;
	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = aShape.count;
	// std::cout<<cShape<<std::endl;
	return cShape;
}

void makeCudaTensor(float *Tensor, TensorShape aShape)
{
	uint64_t offset;
	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);
	for (uint32_t count = 0; count < aShape.count; ++ count) {
		for (uint32_t chIdx = 0; chIdx < aShape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < aShape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < aShape.width; ++ colIdx) {
					offset = count * aShape.channels * aShape.height * aShape.width + chIdx * aShape.height * aShape.width + rowIdx * aShape.width + colIdx;
					Tensor[offset] = dist(random_device);
				}
			}
		}
	}
}

__global__ void convLayer_gpu ( float * input, TensorShape iShape, float * filter, TensorShape fShape, float * bias, float * output, TensorShape oShape, ConvLayerArgs args, uint32_t batchSize) 
{
	int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

	// Computing the indices of output tensor that each thread would be computing in OutputStationary Loop Order.
	int colOut = blockIdx.x * blockDim.x + tx;
	int rowOut = blockIdx.y * blockDim.y + ty;
	int axiOut = blockIdx.z * blockDim.z + tz;
	
	/*
	Two arrays in Shared Memory for Input Activation and Filter Weights. (Note: C is tiled and performed in multiple phases)
	Input Activation = smActH x smActW x tileSize 
	Filter Weights   = blockDim.z x tileSize x fShape.H x fShape.W 
	*/
	extern __shared__ float sharedMemory[];
	uint32_t smActH  = blockDim.y*args.strideH + fShape.height - args.strideH;
	uint32_t smActW  = blockDim.x*args.strideW + fShape.width - args.strideW;
	uint32_t smActD  = blockDim.z;
	float *sharedMemoryAct = sharedMemory;                     
	float *sharedMemoryFil = (float*)&sharedMemoryAct[smActH*smActW*smActD];

	// Outer Batch Loop.
	for(int n = 0; n < batchSize; ++n)
	{
		/*
		Per thread Local Register to store and accumulate partial sums. Initialized with bias.
		Storing in per thread local register reduces the number of global memory writes of output matrix.
		This strategy works sice each thread computes an independent output pixel in output stationary loop order fashion. 
		Optimization: Moved the addition of bias into "last if" condition to prevent thread divergence.
		*/
		float accumulate;
		// if (axiOut < oShape.channels)
			// accumulate = bias[axiOut];
		// else
		accumulate = 0.0;

		/*
		Tiled C Loop. The loop runs for ceil(fShape.channels*1.0/smActD) phases. 
		In each phase, Inputs of depth smActD is loaded into SM, Filters of depth smActD is loaded into SM and output partial sums are accumulated.  
		*/
		for(int ii = 0; ii < ceil(fShape.channels*1.0/smActD); ++ii)
		{
			/*
			Load Input Activations into Shared Memory in a collaboartive manner.
			Each thread block loads smActH x smActW x TileSize dimension of input into SM using tileSize x tileSize x tileSize threads.
			Threads in X and Y dimension have to load more than one element in each phase.
			Therefore, the indices for Z or Channel dimension could be easily computed and it's straight forward. 
			Uses ii for determining which phase the computation is currently in and loads the corresponding values.
			For computing row and column indices of the input activation to be loaded in collaboarative manner, 
			we use the intuituion that "If one vertex of rectangle and its dimensions are known, we could compute other vertices".
			i.e. All threads in a thread block compute their corresponding addresses to load based on the thread(0,0).
			High Level Idea: Load smActH x smActW inputs using tileSize x tileSize threads. (Box Packing Problem)  
			*/
			for(int i = 0; i < smActD; i = i + blockDim.z)
			{	 
				for(int j = 0; j < smActH; j = j + blockDim.y)
				{
					for(int k = 0; k < smActW; k = k + blockDim.x)
					{
						int colInp = (blockIdx.x * blockDim.x)*args.strideW + tx + k - args.padW;
						int rowInp = (blockIdx.y * blockDim.y)*args.strideH + ty + j - args.padH;
						int axiInp = ii*smActD + tz + i;

						int sharedMemCol = tx + k; int sharedMemRow = ty + j;  int sharedMemAxi = tz + i;

						if (sharedMemCol < smActW && sharedMemRow < smActH &&  sharedMemAxi < smActD)
						{	
							int sharedMemoryIndex = sharedMemAxi*smActH*smActW + sharedMemRow*smActW + sharedMemCol;
							uint64_t inputIndex = n*iShape.channels*iShape.height*iShape.width + axiInp*iShape.height*iShape.width + rowInp*iShape.width + colInp;
							if (colInp >= 0 && colInp < iShape.width && rowInp >= 0 && rowInp < iShape.height &&  axiInp >= 0 && axiInp < iShape.channels)
							{
								sharedMemoryAct[sharedMemoryIndex] = input[inputIndex];
							}
							else
							{
								sharedMemoryAct[sharedMemoryIndex] = 0.0;
							}
						}
					}
				}
			}
			/*
			Load Filter Weights into Shared Memory in a collaboartive manner.
			Each thread block loads H x W x TileSize x TileSize dimension of filters into SM using tileSize x tileSize x tileSize threads.
			Threads in X and Y dimension have to load more/less than one element in each phase.
			The indices computation for X or Row, Y or columnn and Z or Channel dimension is same as explained above.
			Since Filter is a 4D tensor, there's one more index that needs to be computed. (M - Number of Filters)
			M is directly related to OShape.Channels and it could be computed as fucntion of blockIdx.z.
			*/
			for(int h = 0; h < blockDim.z; ++h)
			{
				for(int i = 0; i < smActD; i = i + blockDim.z)
				{ 
					for(int j = 0; j < fShape.height; j = j + blockDim.y)
					{
						for(int k = 0; k < fShape.width; k = k + blockDim.x)
						{
							int sharedMemCol = tx + k; int sharedMemRow = ty + j;  int sharedMemAxi = tz + i;
							int axiFilter = ii*smActD + tz + i;
							int fourthDim = blockIdx.z * blockDim.z + h;
							if (sharedMemCol < fShape.width && sharedMemRow < fShape.height &&  sharedMemAxi < fShape.channels)
							{		
								int sharedMemoryIndex = h*smActD*fShape.height*fShape.width + sharedMemAxi*fShape.height*fShape.width + sharedMemRow*fShape.width + sharedMemCol;
								uint64_t filterIndex = fourthDim*fShape.channels*fShape.height*fShape.width + axiFilter*fShape.height*fShape.width + sharedMemRow*fShape.width + sharedMemCol;
								if(blockIdx.z*blockDim.z + h < fShape.count && axiFilter < fShape.channels)
								{
									sharedMemoryFil[sharedMemoryIndex] = filter[filterIndex];
								}
								else
								{
									sharedMemoryFil[sharedMemoryIndex] = 0;

								}		
							}
						}
					}
				}
			}
			/*
			SyncThreads. To wait for all threads to load their input in a collaboartive fashion before starting computation.
			*/
			__syncthreads();
			/*
			Compute Partial Sums.
			We have loaded smActH x smActW x smActD Inputs and filterH x filterW x smActD x TileSize Filters.
			Compute partial sums for all threads in thread block by using three loops of filter dimesnions.
			Because, Each 2D Slice(x and Y) of thread block, uses independent filters. But still they reuse 
			input activations.
			*/
			for(int i = 0; i < smActD; ++i)
			{
				for (int j = 0; j < fShape.height; ++j) 
				{
					for (int k = 0; k < fShape.width; ++k) 
					{
						int sharedMemCol = tx*args.strideW + k;
						int sharedMemRow = ty*args.strideH + j;
						int sharedMemAxi = i; 
						
						int smActIndex = sharedMemAxi*smActH*smActW + sharedMemRow*smActW + sharedMemCol;
						int smFilIndex = (axiOut%blockDim.z)*smActD*fShape.height*fShape.width + i*fShape.height*fShape.width + j*fShape.width +k;

						accumulate += sharedMemoryAct[smActIndex] * sharedMemoryFil[smFilIndex];
					}
				}
			}
			/*
			SyncThreads. To wait for all threads to complete their computation before starting the next phase of computation.
			*/
			__syncthreads();
		}
		/*
		Write to Output.
		If a thread computes valid output, then write the local register to global memory.
		This step can't be done at first, as all threads had to perform collaboarative loading and computation.
		*/
		if(colOut < oShape.width && rowOut < oShape.height && axiOut < oShape.channels)
		{
			uint64_t outIndex = n*oShape.channels*oShape.height*oShape.width + axiOut*oShape.height*oShape.width + rowOut*oShape.width + colOut;
			output[outIndex] = accumulate + bias[axiOut];
		}
	}
}

__global__ void gemmLayer_gpu (float * a, TensorShape aShape, float * b, TensorShape bShape, float * c, TensorShape cShape, GemmLayerArgs args, uint32_t batchSize)
{
	int tx = threadIdx.x; int ty = threadIdx.y;

	int colOut = blockIdx.x * blockDim.x + tx;
	int rowOut = blockIdx.y * blockDim.y + ty;

	extern __shared__ float sharedMemory[];
	float *sharedMemoryMatA = sharedMemory;                     
	float *sharedMemoryMatB = (float*)&sharedMemoryMatA[args.tileH*args.tileH];
	
	// Outer Batch Loop
	for(int i = 0; i < cShape.count; ++i)
	{
		// Outer Channel Loop
		for(int j = 0; j < cShape.channels; ++j)
		{
			/*
			Tiled K loop of N -> C -> H -> W -> K.
			number of phases or folds is computed and for each phase, small tile is loaded into shared memory and 
			partial sums are computed and stored in per thread local register. 
			*/
			int numFolds = ceil((aShape.width*1.0)/args.tileH);
			float accumulate = 0;

			for( int k = 0; k < numFolds; ++k)
			{
				/*
				Matrix A Load Phase:
				Load TileH X TileH from Matrix A into shared Memory using TileH x TileW threads.
				Each thread loads more/less than one value into SM.
				*/
				int rowIndexA = rowOut;
				int colIndexA = 0;
				uint64_t inputAIndex = 0;
				for(int l = 0; l < args.tileH; l += args.tileW)
				{
					int sharedMemColIndex = tx + l;
					colIndexA = k*args.tileH + tx + l;
					if(sharedMemColIndex < args.tileH)
					{
						if(rowIndexA < aShape.height && colIndexA < aShape.width)
						{
							inputAIndex = i*aShape.channels*aShape.height*aShape.width + j*aShape.height*aShape.width + rowIndexA*aShape.width + colIndexA;
							sharedMemoryMatA[ ty*args.tileH + sharedMemColIndex] = a[inputAIndex];
						}
						else
						{
							sharedMemoryMatA[ ty*args.tileH + sharedMemColIndex] = 0.0;
						}
					}
				}
				/*
				Matrix B Load Phase:
				Load TileH X TileW from Matrix B into shared memory using TileH x TileW threads. 
				Each thread loads one memory value into SM.
				*/
				int rowIndexB = k*args.tileH + ty;
				int colIndexB = colOut;
				if(rowIndexB < bShape.height && colIndexB < bShape.width)
				{
					uint64_t inputBIndex = i*bShape.channels*bShape.height*bShape.width + j*bShape.height*bShape.width + rowIndexB*bShape.width + colIndexB;
					sharedMemoryMatB[ ty*args.tileW + tx] = b[inputBIndex];
				}
				else
				{
					sharedMemoryMatB[ ty*args.tileW + tx] = 0.0;

				}
				/*
				SyncThreads: Wait for all threads to load before consuming them.
				*/        
				__syncthreads();
				for(int l = 0; l < args.tileH; ++l)
				{
					accumulate += sharedMemoryMatA[ty*args.tileH + l]*sharedMemoryMatB[l*args.tileW + tx];
				}
				/*
				SyncThreads: Wait for all threads to consume before producing new values.
				*/
				__syncthreads();
			}
			/*
			Write to global memory only if that element computes an useful element.
			*/
			if(colOut < cShape.width && rowOut < cShape.height)
			{
				uint64_t outIndex = i*cShape.channels*cShape.height*cShape.width + j*cShape.height*cShape.width + rowOut*cShape.width + colOut;
				c[outIndex] = accumulate;
			}
		}
	}
}

__global__ 
void poolLayer_gpu (float * input, TensorShape inShape, float * output, TensorShape outShape, PoolLayerArgs args) 
{
	extern __shared__ float sharedMemory[];

	int tx = threadIdx.x; int ty = threadIdx.y;
	int rowOut = blockIdx.y * blockDim.y + ty;
	int colOut = blockIdx.x * blockDim.x + tx;

	int poolH = args.poolH;
	int poolW = args.poolW;
	int strideH = args.strideH;
	int strideW = args.strideW;

	int smHeight = (blockDim.y - 1) * strideH + poolH;
	int smWidth  = (blockDim.x - 1) * strideW + poolW; 

	int outputH = outShape.height;		
	int outputW = outShape.width;
	int outputChannels = outShape.channels;
	int batchSize = outShape.count;

	for(int batch = 0; batch < batchSize; ++batch)
	{
		for(int channel = 0; channel < outputChannels; ++channel)
		{
			for(int i = 0; i < smHeight; i = i + blockDim.y)
			{
				for(int j = 0; j < smWidth; j = j + blockDim.x)
				{
					int rowInp = (blockIdx.y * blockDim.y)*strideH + ty + i;
					int colInp = (blockIdx.x * blockDim.x)*strideW + tx + j;

					int sharedMemRow = ty + i; int sharedMemCol = tx + j;
					if (sharedMemRow < smHeight && sharedMemCol < smWidth)
					{
						if (rowInp >= 0 && rowInp < inShape.height && colInp >= 0 && colInp < inShape.width)
						{
							sharedMemory[sharedMemRow*smWidth + sharedMemCol] = input[batch*inShape.channels*inShape.height*inShape.width + channel*inShape.height*inShape.width + rowInp*inShape.width + colInp];	
						}
						else
						{
							sharedMemory[sharedMemRow*smWidth + sharedMemCol] = 0.0;
						}
					}
				}
			}
			__syncthreads();
			if (rowOut < outputH && colOut < outputW)
			{
				float poolVal = 0;
				switch (args.opType)
				{
					case PoolOp::MaxPool:
						poolVal =  -1000000001;
						break;
					case PoolOp::AvgPool:
						poolVal = 0;
						break;
					case PoolOp::MinPool:
						poolVal = 1000000001;
						break;
				}
				for (int poolRow = 0; poolRow < poolH; ++ poolRow)
				{
					for (int poolCol = 0; poolCol < poolW; ++ poolCol)
					{
						int sharedMemRow = ty*strideH + poolRow; 
						int sharedMemCol = tx*strideW + poolCol;

						float pixelVal = sharedMemory[sharedMemRow*smWidth + sharedMemCol];
						switch (args.opType)
						{
							case PoolOp::MaxPool:
								poolVal = fmax(poolVal, pixelVal);
								break;
							case PoolOp::AvgPool:
								poolVal = poolVal + pixelVal;
								break;
							case PoolOp::MinPool:
								poolVal = fmin(poolVal, pixelVal);
								break;
						}
					}
				}
				switch (args.opType)
				{
					case PoolOp::AvgPool:
						poolVal = poolVal/(poolH*poolW);
						break;
				}
				output[batch*outShape.channels*outShape.height*outShape.width + channel*outShape.height*outShape.width + rowOut*outShape.width + colOut] = poolVal;
			}
			__syncthreads();			
		}
	}
}


float* Convolution(float* act, TensorShape actShape, TensorShape filterShape, ConvLayerArgs args)
{
	float *filter, *bias, *output;
	TensorShape oShape = ComputeConvOutput(actShape, filterShape, args);
	
	cudaMallocManaged(&filter, tensorSize(filterShape)*sizeof(float));
	cudaMallocManaged(&bias,   oShape.channels*sizeof(float));
	cudaMallocManaged(&output, tensorSize(oShape)*sizeof(float));
	makeCudaTensor(filter, filterShape);
	TensorShape biasShape = {1, 1, 1, oShape.channels};
	makeCudaTensor(bias, biasShape);

	int tileSize = 4;
	dim3 blockDim(tileSize,tileSize, tileSize);
	dim3 gridDim(ceil(oShape.height*1.0/tileSize), ceil(oShape.width*1.0/tileSize), ceil(oShape.channels*1.0/tileSize));

	// SharedMemory Size of Input Activations
	int channelBlocking = tileSize;
	uint32_t smActH  = tileSize*args.strideH + filterShape.height - args.strideH;
	uint32_t smActW  = tileSize*args.strideW + filterShape.width - args.strideW;
	uint32_t smActD  = channelBlocking;
	// SharedMemory Size of Filters
	uint32_t smFilH  = filterShape.height;
	uint32_t smFilW  = filterShape.width;
	uint32_t smFilD  = channelBlocking;
	
	size_t dynamicSharedMemsize = smActH*smActW*smActD*sizeof(float) + tileSize*smFilH*smFilW*smFilD*sizeof(float);

	convLayer_gpu<<<gridDim, blockDim, dynamicSharedMemsize>>>(act, actShape, filter, filterShape, bias, output, oShape, args, oShape.count);
	cudaDeviceSynchronize();

	cudaFree(bias);
	cudaFree(filter);

	return output;
}

float* Pooling(float* act, TensorShape actShape, PoolLayerArgs args)
{
	float *output;
	TensorShape oShape = ComputePoolOutput(actShape, args);
	
	cudaMallocManaged(&output, tensorSize(oShape)*sizeof(float));

	int tileSize = 4;
	dim3 blockDim(tileSize, tileSize);
	dim3 gridDim(ceil(oShape.height*1.0/tileSize), ceil(oShape.width*1.0/tileSize));

	uint32_t smHeight = (tileSize - 1)*args.strideH + args.poolH;
	uint32_t smWidth  = (tileSize - 1)*args.strideW + args.poolW;
	size_t dynamicSharedMemsize = smHeight*smWidth*sizeof(float);

	poolLayer_gpu<<<gridDim,blockDim, dynamicSharedMemsize>>>(act, actShape, output, oShape, args);
	cudaDeviceSynchronize();

	return output;
}

float* FullyConv(float* act, TensorShape actShape, TensorShape filterShape, GemmLayerArgs args)
{
	float *filter, *output;
	TensorShape oShape = ComputeFCOutput(actShape, filterShape);

	cudaMallocManaged(&filter, tensorSize(filterShape)*sizeof(float));
	makeCudaTensor(filter, filterShape);

	cudaMallocManaged(&output, tensorSize(oShape)*sizeof(float));

	dim3 blockDim(args.tileW, args.tileH);
	dim3 gridDim(ceil((oShape.width*1.0)/args.tileW), ceil((oShape.height*1.0)/args.tileH));
	size_t dynamicSharedMemsize = args.tileH*args.tileH*sizeof(float) + args.tileH*args.tileW*sizeof(float);
	
	gemmLayer_gpu<<<gridDim, blockDim, dynamicSharedMemsize>>>(act, actShape, filter, filterShape, output, oShape, args, oShape.count);
	cudaDeviceSynchronize();
	cudaFree(filter);

	return output;
}

int runGpuAlexNet (int batch)
{
	// Layerwise Parameters
	int batchSize = batch;
	TensorShape InputTensorShape = {batchSize, 3, 227, 227};	
	TensorShape Conv1FilterShape = {96,3, 11,11};
	ConvLayerArgs Conv1Args = {0, 0, 4, 4, true};

	PoolLayerArgs MaxPool1Args = {PoolOp::MaxPool, 3, 3, 2, 2};

	TensorShape Conv2FilterShape = {256,96, 5, 5};
	ConvLayerArgs Conv2Args = {2, 2, 1, 1, true};

	PoolLayerArgs MaxPool2Args = {PoolOp::MaxPool, 3, 3, 2, 2};	

	TensorShape Conv3FilterShape = {384,256, 3, 3};
	ConvLayerArgs Conv3Args = {1, 1, 1, 1, true};
	
	TensorShape Conv4FilterShape = {384,384, 3, 3};
	ConvLayerArgs Conv4Args = {1, 1, 1, 1, true};

	TensorShape Conv5FilterShape = {256,384, 3, 3};
	ConvLayerArgs Conv5Args = {1, 1, 1, 1, true};

	PoolLayerArgs MaxPool3Args = {PoolOp::MaxPool, 3, 3, 2, 2};

	TensorShape FC1FilterShape = {1, 1, 9216, 4096};
	TensorShape FC2FilterShape = {1, 1, 4096, 4096};
	TensorShape FC3FilterShape = {1, 1, 4096, 1000};
	GemmLayerArgs args = {8, 8, 1};

	/*
	Conv -> ReLu -> MaxPool -> Conv -> ReLu -> MaxPool -> Conv -> ReLu -> Conv -> ReLu -> Conv -> ReLu -> MaxPool
	-> FC -> FC -> FC 
	)
	*/
	TensorShape oShape, actShape;
	oShape = ComputeConvOutput(InputTensorShape, Conv1FilterShape, Conv1Args);
	oShape = ComputePoolOutput(oShape, MaxPool1Args);
	oShape = ComputeConvOutput(oShape, Conv2FilterShape, Conv2Args);
	oShape = ComputePoolOutput(oShape, MaxPool2Args);
	oShape = ComputeConvOutput(oShape, Conv3FilterShape, Conv3Args);
	oShape = ComputeConvOutput(oShape, Conv4FilterShape, Conv4Args);
	oShape = ComputeConvOutput(oShape, Conv5FilterShape, Conv5Args);
	oShape = ComputePoolOutput(oShape, MaxPool3Args);

	oShape.width = oShape.channels*oShape.width*oShape.height;
	oShape.height =  batchSize;
	oShape.channels = 1;
	oShape.count = 1;

	oShape = ComputeFCOutput(oShape, FC1FilterShape);
	oShape = ComputeFCOutput(oShape, FC2FilterShape);
	oShape = ComputeFCOutput(oShape, FC3FilterShape);

	float *InputTensor;
	cudaMallocManaged(&InputTensor, tensorSize(InputTensorShape)*sizeof(float));
	makeCudaTensor(InputTensor, InputTensorShape);

	float *act = Convolution(InputTensor, InputTensorShape, Conv1FilterShape, Conv1Args);
	actShape = ComputeConvOutput(InputTensorShape, Conv1FilterShape, Conv1Args);

	act = Pooling(act, actShape, MaxPool1Args);
	actShape = ComputePoolOutput(actShape, MaxPool1Args);

	act = Convolution(act, actShape, Conv2FilterShape, Conv2Args);
	actShape = ComputeConvOutput(actShape, Conv2FilterShape, Conv2Args);
	
	act = Pooling(act, actShape, MaxPool2Args);
	actShape = ComputePoolOutput(actShape, MaxPool2Args);
	
	act = Convolution(act, actShape, Conv3FilterShape, Conv3Args);
	actShape = ComputeConvOutput(actShape, Conv3FilterShape, Conv3Args);

	act = Convolution(act, actShape, Conv4FilterShape, Conv4Args);
	actShape = ComputeConvOutput(actShape, Conv4FilterShape, Conv4Args);

	act = Convolution(act, actShape, Conv5FilterShape, Conv5Args);
	actShape = ComputeConvOutput(actShape, Conv5FilterShape, Conv5Args);

	act = Pooling(act, actShape, MaxPool3Args);
	actShape = ComputePoolOutput(actShape, MaxPool3Args);

	actShape.width = actShape.channels*actShape.width*actShape.height;
	actShape.height =  batchSize;
	actShape.channels = 1;
	actShape.count = 1;
	
	act = FullyConv(act, actShape, FC1FilterShape, args);
	actShape = ComputeFCOutput(actShape, FC1FilterShape);

	act = FullyConv(act, actShape, FC2FilterShape, args);
	actShape = ComputeFCOutput(actShape, FC2FilterShape);

	act = FullyConv(act, actShape, FC3FilterShape, args);
	actShape = ComputeFCOutput(actShape, FC3FilterShape);

	return 0;
}
