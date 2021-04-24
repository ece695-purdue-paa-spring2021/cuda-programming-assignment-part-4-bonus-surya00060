
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




int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	
	std::cout << "Lazy, you are! ... ";
	std::cout << "Filter pixels, you must! ... ";

	return 0;
}

int medianFilter_gpu (uint8_t inPixels, ImageDim imgDim, 
	uint8_t outPixels, MedianFilterArgs args) {

	return 0;
}


int runGpuConv (int argc, char ** argv) {

	TensorShape iShape = AlexL1_InShape;
	TensorShape fShape = AlexL1_FilterShape;
	ConvLayerArgs convArgs = AlexL1_ConvArgs;

	std::cout << "Evaluate convolution : \n";
	std::cout << "Input : " << iShape << " \n";
	std::cout << "Filter : " << fShape << " \n";

	TensorShape oShape;

	uint64_t errorCount = evaluateGpuConv(iShape, fShape, oShape, convArgs);
	std::cout << "Found " << errorCount << " / " << tensorSize(oShape) << " errors \n";
	return 0;
}

uint64_t evaluateGpuConv (TensorShape iShape, TensorShape fShape, 
	TensorShape & oShape, ConvLayerArgs args) {

	uint64_t errorCount = 0;

	//	STUDENT: Add code here

	#ifndef CONV_CHECK_DISABLE
		//	STUDENT: Verify number of errors in ouput matrix generated by convLayer_gpu
		//	STUDENT: Compare results with CPU output
		//	STUDENT: Return error count


	#endif

	return errorCount;
}

int convLayer_gpu ( float * input, TensorShape iShape, 
	float * filter, TensorShape fShape, 
	float * bias, float * output, TensorShape & oShape, 
	ConvLayerArgs & args, uint32_t batchSize) {

	return 0;
}


int runGpuGemm (int argc, char ** argv) {

	evaluateGpuGemm();
	return 0;
}

int evaluateGpuGemm () {

	return 0;
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
	print(oShape);
	return oShape;
}

TensorShape ComputePoolOutput(TensorShape iShape, PoolLayerArgs args)
{
	TensorShape oShape;
	oShape.height 	= (iShape.height - args.poolH) / args.strideH + 1;
	oShape.width	= (iShape.width  - args.poolW) / args.strideW + 1;
	oShape.channels	= iShape.channels;
	oShape.count 	= iShape.count;
	print(oShape);
	return oShape;
}

TensorShape ComputeFCOutput(TensorShape aShape, TensorShape bShape)
{
	TensorShape cShape;
	cShape.height = aShape.height;
	cShape.width = bShape.width;
	cShape.channels = aShape.channels;
	cShape.count = aShape.count;
	print(oShape);
	return cShape;
}
int runGpuAlexNet (int argc, char ** argv)
{
	int batchSize = 1;
	TensorShape InputTensor = {batchSize, 3, 227, 227};	
	TensorShape Conv1FilterShape = {96,3, 11,11};
	ConvLayerArgs Conv1Args = {0, 0, 4, 4, true};

	PoolLayerArgs MaxPool1Args = {MaxPool, 3, 3, 2, 2};

	TensorShape Conv2FilterShape = {256,96, 5, 5};
	ConvLayerArgs Conv2Args = {2, 2, 1, 1, true};

	PoolLayerArgs MaxPool2Args = {MaxPool, 3, 3, 2, 2};	

	TensorShape Conv3FilterShape = {384,256, 3, 3};
	ConvLayerArgs Conv3Args = {1, 1, 1, 1, true};
	
	TensorShape Conv4FilterShape = {384,384, 3, 3};
	ConvLayerArgs Conv4Args = {1, 1, 1, 1, true};

	TensorShape Conv5FilterShape = {256,384, 3, 3};
	ConvLayerArgs Conv5Args = {1, 1, 1, 1, true};

	PoolLayerArgs MaxPool3Args = {MaxPool, 3, 3, 2, 2};

	TensorShape FC1FilterShape = {1, 1, 9216, 4096};
	TensorShape FC2FilterShape = {1, 1, 4096, 4096};
	TensorShape FC3FilterShape = {1, 1, 4096, 1000};
	GemmLayerArgs args = {8, 8, 1};

	/*
	Conv -> ReLu -> MaxPool -> Conv -> ReLu -> MaxPool -> Conv -> ReLu -> Conv -> ReLu -> Conv -> ReLu -> MaxPool
	-> FC -> FC -> FC 
	)
	*/
	TensorShape oShape;
	oShape = ComputeConvOutput(InputTensor, Conv1FilterShape, Conv1Args);
	oShape = ComputePoolOutput(oShape, MaxPool1Args);
	oShape = ComputeConvOutput(oShape, Conv2FilterShape, Conv2Args);
	oShape = ComputePoolOutput(oShape, MaxPool2Args);
	oShape = ComputeConvOutput(oShape, Conv3FilterShape, Conv3Args);
	oShape = ComputeConvOutput(oShape, Conv4FilterShape, Conv4Args);
	oShape = ComputeConvOutput(oShape, Conv5FilterShape, Conv5Args);
	oShape = ComputePoolOutput(oShape, MaxPool3Args);
	oShape = ComputeFCOutput(oShape, FC1FilterShape);
	oShape = ComputeFCOutput(oShape, FC2FilterShape);
	oShape = ComputeFCOutput(oShape, FC3FilterShape);

	return 0;
}