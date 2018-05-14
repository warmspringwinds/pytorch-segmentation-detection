#include <caffe2/core/init.h>
#include <caffe2/core/workspace.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/blob.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/context_gpu.h>
#include "caffe2/utils/smart_tensor_printer.h"

#include <chrono>
#include <cuda_runtime.h>


using namespace caffe2;
using namespace std;
using namespace std::chrono;


int main(int argc, char** argv) 
{
        
    string init_net_filename = "../init_net.pb";
    string predict_net_filename = "../predict_net.pb";
    
    Workspace workspace;

    //>>> data = np.random.rand(16, 100).astype(np.float32)
    std::vector<float> data( 3 * 512 * 512);
    
    for (auto& v : data)
    {
        v = (float) 255;
    }
    
    auto cpu_tensor = TensorCPU({1, 3, 512, 512}, data, NULL);
    
    // Transferring to GPU
    auto gpu_tensor = TensorCUDA(cpu_tensor);
    
    // Creating a named input blob that the network expects and writing
    // our tensor value into that blob
    auto gpu_tensor_workspace = workspace.CreateBlob("actual_input_1")->GetMutable<TensorCUDA>();
    gpu_tensor_workspace->ResizeLike(gpu_tensor);
    gpu_tensor_workspace->ShareData(gpu_tensor);
    
    // NetDef is a protobuf object
    // We read protobuf files into these objects
    NetDef init_net, predict_net;
    
    // Reading the 
    CAFFE_ENFORCE( ReadProtoFromFile(init_net_filename, &init_net) );
    CAFFE_ENFORCE( ReadProtoFromFile(predict_net_filename, &predict_net) );
    
    init_net.mutable_device_option()->set_device_type(CUDA);
    predict_net.mutable_device_option()->set_device_type(CUDA);
    
    CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
    CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));
    
    workspace.CreateNet(predict_net);
    
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;
    
    int number_of_iterations = 100;
    int overall_miliseconds_count = 0;
    
    for (int i = 0; i < number_of_iterations; ++i)
    {

        t1 = high_resolution_clock::now();
        
        workspace.RunNet(predict_net.name());
        
        cudaDeviceSynchronize();

        t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

        overall_miliseconds_count += duration;

    }
    
    // Average execution time: 29.59 ms -- on our machine
    cout << "Average execution time: " << overall_miliseconds_count / float(number_of_iterations) << " ms" << endl;
     
    return 0;
}