#include <vector>
#include <iostream>
#include <fstream>
using std::vector;
using std::endl;
using std::cout;

long merge(vector<int> &array, int left, int middle, int right) {
	long inversionCount = 0;
	int leftSize = middle - left + 1; // Extra one for odd
	int leftCopy[leftSize];
	for (int i = 0; i < leftSize; ++i){
		leftCopy[i] = array[left + i];
	}

	int rightSize = right - middle;
	int rightCopy[rightSize];
	for (int i = 0; i < rightSize; ++i){
		rightCopy[i] = array[middle+1+i];
	}

	int leftIndex = 0, rightIndex = 0;
	int arIndex = left;
	while (leftIndex < leftSize || rightIndex < rightSize){
		if (((leftIndex < leftSize) && (leftCopy[leftIndex] < rightCopy[rightIndex]))
			|| (rightIndex >= rightSize)){
			array[arIndex] = leftCopy[leftIndex];
			++leftIndex;
		} else {
			array[arIndex] = rightCopy[rightIndex];
			++rightIndex;
			inversionCount += (leftSize - leftIndex);
		}
		++arIndex;
	}
	return inversionCount;
} 

long mergeSort (vector<int> &v, int left, int right) {
	if (left < right){
		int mid = (left/2 + right/2);
		return mergeSort(v,left,mid) + mergeSort(v,mid+1,right) + merge(v, left, mid, right);
	}
	return 0;
}

void printArray(vector<int> array){
	for (auto elem:array){
		std::cout << elem << " ";
	}
	std::cout << std::endl;
} 

// Requires a list of numbers from file
int main (int argc, char * argv[]) {
	vector<int> array;
	if (argc > 1) {
		const std::string& file = argv[1];
		std::ifstream in {file};
		int a;
		while (in >> a){
			array.emplace_back(a);
		}
	} else {
		std::cout << "please provide a file" << std::endl; 
	}

	printArray(array);
	std::cout << "Inversions: " << mergeSort(array, 0, array.size() - 1) << std::endl;
	printArray(array);
}



