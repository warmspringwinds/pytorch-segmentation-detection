#include <caffe2/core/init.h>
#include <caffe2/core/workspace.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/blob.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/context_gpu.h>
#include "caffe2/utils/smart_tensor_printer.h"


#include <cuda_runtime.h>


using namespace caffe2;
using namespace std;


int main(int argc, char** argv) 
{
    
   int width = 512;
   int height = 512;

    // Dummy GPU image -- RGBA
    std::vector<unsigned char> image(4 * width * height);
    
    for (size_t y = 0; y < height; ++y) 
    {
        for (size_t x = 0; x < width; ++x) 
        {
            size_t idx = y * width + x;
            
            // 35 -- is a '$' ascii symbol
            image[ idx * 4 + 0] = 36;
            image[ idx * 4 + 1] = 36;
            image[ idx * 4 + 2] = 36;
            image[ idx * 4 + 3] = 36;

        }
    }

    //Load the dummy image to GPU
    unsigned char * cuda_pointer;
    cudaMalloc(&cuda_pointer, 4 * width * height * sizeof(unsigned char));
    cudaMemcpy(cuda_pointer, image.data(), sizeof(unsigned char) * 4 * width * height, cudaMemcpyHostToDevice);
    
    
    auto size = vector<int>({1, 4, 512, 512});
    auto gpu_tensor_char = TensorCUDA(size);
    
    // Read more here and in the documentation of Caffe2
    //https://github.com/caffe2/caffe2/issues/1004
    gpu_tensor_char.ShareExternalPointer(cuda_pointer, 4 * width * height);
    
    // TODO: find out how to do conversion from char to float on gpu using caffe2 API
    // wasn't really obvious if it's possible to do without using caffe2 ops.
    
    // Transferring the tensor to cpu
    auto cpu_tensor_char = TensorCPU(gpu_tensor_char);
    
    // Should print out a lot of money '$'
    SmartTensorPrinter::PrintTensor(cpu_tensor_char);
    
    return 0;
}