#include "main.h"
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <vector>
#include<iterator>
#include<numeric>
#include <torch/torch.h>
#include "cifar10.h"

using namespace nvinfer1;
using namespace std;
using namespace torch;

// logger required by tensorrt
class Logger: public ILogger
{
        void log(Severity severity, const char* msg) override
        {
                if(severity!= Severity::kINFO)
                        cout<<msg<<endl;
        }

}gLogger;


// builds a inference engine from onnx model
// configured to accept  dynamic batch size
// ARGUMENTS:
//      string  file_name: The path to the model
//      int optimal_bsize: The batchsize for which to optimize the engine
ICudaEngine*  build_engine_from_onnx( string file_name, int optimal_bsize)
{

        long  max_workspace_size = 1*(1<<30);
        int max_batch_size = 128;

        IBuilder* builder = createInferBuilder(gLogger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    

        // define parser for onnx files, and parse file given as  argument
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        parser->parseFromFile(std::string(file_name).c_str(), 3); 
        for (int i = 0; i < parser->getNbErrors(); ++i)
                std::cout << parser->getError(i)->desc() << std::endl;

        

        // create build config object
        // build engine based on config
        IBuilderConfig*  config = builder->createBuilderConfig();
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{1, 3, 32 ,32});
        profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{optimal_bsize, 3, 32, 32});
        profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{512, 3, 32 ,32});    
        config->addOptimizationProfile(profile);

        builder->setMaxBatchSize(max_batch_size);
        config->setMaxWorkspaceSize(max_workspace_size);

        config->setFlag(BuilderFlag::kFP16);
        //builder->setFp16Mode(builder->platformHasFastFp16());

        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
 

        // free objects 
        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
    
        return engine;

        // write  serialized model to disk 
        // COMMENT THE return engine ABOVE TO WRITE MODEL TO DISK
        IHostMemory *serializedModel = engine->serialize();

        ofstream ofs("serialized_engine.trt", std::ios::out | std::ios::binary);
        ofs.write((char*)(serializedModel ->data()), serializedModel ->size());
        ofs.close();
        return engine;

}

// this function is not used anywhere, it is just a inference function

void launchInference(ICudaEngine* engine, IExecutionContext* context, cudaStream_t stream, vector<float> & inputTensor, vector<float> & outputTensor, void** bindings, int batchSize)
{
        int inputIndex = engine->getBindingIndex("input");
        int outputIndex = engine->getBindingIndex("output");

        cudaMemcpyAsync(bindings[inputIndex], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueue(batchSize,bindings, stream, nullptr);
        cudaMemcpyAsync(outputTensor.data(), bindings[outputIndex], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
}



void latency_test(IExecutionContext* context,ICudaEngine* engine, int batch_size)
{

        void* bindings[2]{0};

        // load CIFAR test data and normalize images
        string CIFAR_data_path = "../data/cifar-10-batches-bin";
        auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
                .map(torch::data::transforms::Normalize<>( {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225} )).map(torch::data::transforms::Stack<>());

        auto num_test_samples = test_dataset.size().value();
        // define loader for test  data of given batch size
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);
        
        
        // declare buffers needed for inference
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        vector<float>  inputTensor;
        vector<float> outputTensor;

        // get index of input and output
        int inputIndex = engine->getBindingIndex("input");
        int outputIndex = engine->getBindingIndex("output");

        // resize buffers based on dimension of input
        size_t input_size, output_size;
        Dims baseDims = context->getBindingDimensions(inputIndex);
        baseDims.d[0] = batch_size;
        context->setBindingDimensions(inputIndex, baseDims);
        Dims dims{engine->getBindingDimensions(inputIndex)};
        input_size = accumulate(dims.d+1, dims.d + dims.nbDims,batch_size , multiplies<int>());
        cudaMalloc(&bindings[inputIndex], input_size * sizeof(float));
        
        Dims dims1{engine->getBindingDimensions(outputIndex)};
        output_size = accumulate(dims1.d+1, dims1.d + dims1.nbDims,batch_size , multiplies<int>());
        outputTensor.resize(output_size);
        cudaMalloc(&bindings[outputIndex], output_size * sizeof(float));
        
        // for measuring latency
        cudaEvent_t starttime;
        cudaEvent_t endtime;
        cudaEventCreate(&starttime);
        cudaEventCreate(&endtime); 

        // The actual inference stage, loop through batches
        float count = 0,correct = 0,accuracy = 0;
        double TotalTime;
        for (torch::data::Example<>& batch : *test_loader) 
        {
                // measuring for 4000 iterations is enough
                if(count==5000)
                        break;
                float elapsedTime;
                // perform inference and record latency
                cudaEventRecord(starttime, stream); 
                cudaMemcpyAsync(bindings[inputIndex], batch.data.data_ptr(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
                context->enqueue(batch_size,bindings, stream, nullptr);
                cudaMemcpyAsync(outputTensor.data(), bindings[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(endtime, stream);
                cudaStreamSynchronize(stream);

                cudaEventElapsedTime(&elapsedTime, starttime, endtime);
                TotalTime += elapsedTime;

                count++;        
                // find the accuracy(only for batch_size=1 since accuracy is independent of batch_size)
                if(batch_size==1)
                {
                        int predicted_class = distance(outputTensor.begin(), max_element(outputTensor.begin(), outputTensor.end()));
                        int actual_class = batch.target[0].item<int64_t>();
                        if(predicted_class == actual_class)
                                correct++;
                        accuracy = correct/count;

                        cout<<count<<" accuracy is "<<accuracy<<" =  ( "<<correct<<"/"<<count<<" )"<<endl;
                }
                // write latency
                
                cout<<" latency is "<<elapsedTime << endl;
        }
        
        cout<<"------------------------------------------------------------------------------------------------------"<<endl;
        if(batch_size==1)
                cout<<"ACCURACY IS "<<accuracy<<endl;
        cout<<"AVERAGE LATENCY for batch size "<< batch_size<<" is "<<TotalTime/count<<endl;

        // free memory
        context->destroy();
        cudaStreamDestroy(stream);
        cudaEventDestroy(starttime);
        cudaEventDestroy(endtime);
        engine->destroy();
        for (void* ptr : bindings)
                cudaFree(ptr);
}




int main(int argc, char** argv)
{  
        
         
        //-------------------------------------------CREATE ENGINE FROM ONNX FILE---------------------------
        
        int batch_size=1;
        // no arguments given assume batch_size=1
        if(argc==1)
        {
                cout<<"Since no command line arguments were given taking precision as FP16 and batch_size as 1"<<endl;
                cout<<"Please read README.md for more info on the command line arguments"<<endl;
                batch_size=1;
        }
        if(argc==2)
        {
                batch_size = stoi(argv[1]);

        }
  
        cout<<"precision: FP16 "<<" batch_size: "<<batch_size<<endl;

        ICudaEngine *engine = build_engine_from_onnx(string("../data/unet.onnx"),batch_size);
        IExecutionContext *context = engine->createExecutionContext();

        //-------------------------------------------PERFORM LATENCY TEST---------------------------
        latency_test(context,engine,batch_size);
        return 0;
        
 
 }
