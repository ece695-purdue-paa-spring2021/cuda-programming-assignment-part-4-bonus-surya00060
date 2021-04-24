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

	std::cout << "ECE 695 - Lab 4 Bonus\n";
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


