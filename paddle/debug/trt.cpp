#include<iostream>
#include<fstream>
#include<memory>
#include<string>
#include <algorithm>
#include <random>

//#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "cuda_runtime.h"

using namespace nvinfer1;

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t) { if(t) t->destroy(); }
};

template <typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T> >;


class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override {
        //if (severity <= Severity::kWARNING) {
            std::cout<<"log:" << msg << std::endl;
        //}
    }
};

int main(int argc, char* argv[]) {

    if(argc < 2) {
        std::cout<<"error"<<std::endl;
        return -1;
    }
    std::ifstream engineFile(std::string(argv[1]), std::ios::binary);
    if (!engineFile){
        std::cout << "Error opening engine file: " << argv[1] << std::endl;
        return -1;
    }

    int batch = 1;
    if(argc>2) {
        batch = std::atoi(argv[2]);
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << argv[1] << std::endl;
        return -1;
    }

    Logger m_logger;

    TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(m_logger));
    if (!runtime) {
        std::cout<<"creat runtime error"<<std::endl;
        return -1;
    }
    std::cout<<engineData.size()<<std::endl;
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize));
    if (!mEngine) {
        std::cout<<"create mEngine error" << std::endl;
        return -1;
    }

    TrtUniquePtr<nvinfer1::IExecutionContext> mContext(mEngine->createExecutionContext());
    //auto input_idx = mEngine->getBindingIndex("images");
    //std::cout<<input_idx<<std::endl;
    auto dims = mEngine->getBindingDimensions(0);
    std::cout<<dims.d[0]<<"_"<<dims.d[1]<<"_"<<dims.d[2]<<"_"<<dims.d[3]<<std::endl;

    //cudaStream_t stream = nullptr;
    //cudaStreamCreate(&stream);

    std::vector<float> input;
    input.resize(3*224*224);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    std::generate(input.begin(), input.end(),
        [&dis, &gen] () mutable {
            return dis(gen);
        });

    std::cout<<std::endl;

while(1) {
    std::cin>>batch;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    float* input_device = nullptr;
    std::vector<float> output;
    int out_dim = 512;
    output.resize(batch*out_dim);
    float* output_device = nullptr;

    nvinfer1::Dims4 inputDims = {batch, dims.d[1], dims.d[2], dims.d[3]};
    mContext->setBindingDimensions(0, inputDims);
    if(!mContext->allInputDimensionsSpecified()) {
        std::cout<<"Some input dimension is not specified!\n";

    }

    cudaMalloc(&input_device, batch*3*224*224*sizeof(float));
    cudaMalloc(&output_device, batch*512*sizeof(float));

    std::vector<float> input_batch;
    for(int i=0;i<batch;i++) {
        auto pos = input_batch.begin()+i*3*224*224;
        input_batch.insert(pos, input.begin(), input.end());
    }
    std::cout<<cudaMemcpyAsync(input_device, input_batch.data(), batch*3*224*224*sizeof(float),
          cudaMemcpyHostToDevice, stream)<<std::endl;

    float *binds[] = {input_device, output_device};
    //mContext->executeV2((void**)binds);
    auto status = mContext->enqueueV2((void**)binds, stream, nullptr);

    if(!status) std::cout<<"Something is wrong in inference!\n";

    std::cout<<status<<std::endl;

    std::cout<<cudaMemcpyAsync(output.data(), output_device, batch * out_dim * sizeof(float),
     cudaMemcpyDeviceToHost, stream)<<std::endl;

    cudaStreamSynchronize(stream);
    cudaFree(input_device);
    cudaFree(output_device);
    cudaStreamDestroy(stream);

    for(int j=0;j<batch;j++) {
    for(int i=0;i<out_dim;i++) {
        std::cout<<output[j*out_dim + i]<<",";
    }
     std::cout<<std::endl;
   }

}
    return 0;
}
