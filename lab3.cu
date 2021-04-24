/**
 * @file lab3.cpp
 * @author Abhishek Bhaumick (abhaumic@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2021-01-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include "lab3.cuh"
#include "cpuLib.h"
#include "cudaLib.cuh"

int main(int argc, char** argv) {
	std::string str;
	int choice;

	std::cout << "ECE 695 - Lab 2 \n";
	std::cout << "Select application: \n";
	std::cout << "  1 - CPU SAXPY \n";
	std::cout << "  2 - GPU SAXPY \n";
	std::cout << "  3 - CPU Monte-Carlo Pi \n";
	std::cout << "  4 - GPU Monte-Carlo Pi \n";
	std::cout << "  5 - Bytes-Image File Test \n";
	std::cout << "  6 - Median Filter CPU \n";
	std::cout << "  7 - Median Filter GPU \n";
	std::cout << "  8 - Pool CPU \n";
	std::cout << "  9 - Pool GPU \n";
	std::cout << "  12 - Convolution on CPU \n";
	std::cout << "  13 - Convolution on GPU \n";
	std::cout << "  14 - Matrix Multiply on CPU \n";
	std::cout << "  15 - Matrix Multiply on GPU \n";

	std::cin >> choice;

	std::cout << "\n";
	std::cout << "Choice selected - " << choice << "\n\n";

	PoolLayerArgs poolArgs;
	MedianFilterArgs filArgs;
	TensorShape inShape;

	std::cout << "Running AlexNet on GPU! \n\n";
	runGpuAlexNet(argc, argv);	
	std::cout << "\n\n ... Done!\n";
	
	return 0;
}

int testLoadBytesImage(std::string filePath) {
	ImageDim imgDim;
	uint8_t * imgData;
	int bytesRead = loadBytesImage(filePath, imgDim, &imgData);
	int bytesExpected = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;
	if (bytesRead != bytesExpected) {
		std::cout << "Read Failed - Insufficient Bytes - " << bytesRead 
			<< " / "  << bytesExpected << " \n";
		return -1;
	}
	std::cout << "Read Success - " << bytesRead << " Bytes \n"; 
	return 0;
}


