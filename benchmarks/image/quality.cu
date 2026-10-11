#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/random.h>
#include <stbi/stbi_wrapper.h>
#include <stbi/stb_image_write.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace tcnn;
using P = network_precision_t;
using Clock = std::chrono::steady_clock;
using TrainerT = Trainer<float,P,P>;
std::shared_ptr<TrainerT> make_trainer(std::shared_ptr<NetworkWithInputEncoding<P>> network,const json& config,uint32_t seed) {
    return std::make_shared<TrainerT>(network,
        std::shared_ptr<Optimizer<P>>{create_optimizer<P>(config.value("optimizer",json::object()))},
        std::shared_ptr<Loss<P>>{create_loss<P>(config.value("loss",json{{"otype","RelativeL2"}}))},seed);
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
void write_json(const std::filesystem::path& path, const json& value) {
    std::ofstream file(path); file << value.dump(2); if (!file) throw std::runtime_error("JSON write failed");
}
template<class T> void write_raw(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream file(path,std::ios::binary); file.write((const char*)values.data(),values.size()*sizeof(T));
    if (!file) throw std::runtime_error("Binary write failed");
}

__global__ void sample_pixels(uint32_t n, uint32_t width, uint32_t height, cudaTextureObject_t texture, const float* coordinates, float* targets) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    // Original image benchmark: uniform inputs, nearest-pixel labels via texture.
    float x=(roundf(coordinates[2*i]*width-0.5f)+0.5f)/width;
    float y=(roundf(coordinates[2*i+1]*height-0.5f)+0.5f)/height;
    float4 value=tex2D<float4>(texture,x,y);
    targets[3*i]=value.x; targets[3*i+1]=value.y; targets[3*i+2]=value.z;
}

struct Image {
    int width,height;
    uint32_t pixels,padded;
    std::vector<float> linear,coords;
    std::vector<uint8_t> reference;
    GPUMemory<float> device,grid;
    GPUMemory<uint8_t> reference_device;
    cudaTextureObject_t texture=0;
    Image(const std::string& path) {
        float* source=load_stbi(&width,&height,path.c_str());
        pixels=width*height; padded=next_multiple(pixels,128u);
        linear.assign(source,source+pixels*4); free(source);
        reference.resize(pixels*3); coords.resize(padded*2);
        for (uint32_t i=0;i<padded;++i) {
            uint32_t p=std::min(i,pixels-1);
            coords[2*i]=(p%width+0.5f)/width; coords[2*i+1]=(p/width+0.5f)/height;
            if (i<pixels) for (int c=0;c<3;++c)
                reference[3*i+c]=(uint8_t)std::lround(std::pow(std::max(0.0f,linear[4*i+c]),1.0f/2.2f)*255.0f);
        }
        device.resize(linear.size()); device.copy_from_host(linear.data());
        grid.resize(coords.size()); grid.copy_from_host(coords.data());
        reference_device.resize(reference.size()); reference_device.copy_from_host(reference.data());
        cudaResourceDesc resource{}; resource.resType=cudaResourceTypePitch2D;
        resource.res.pitch2D.devPtr=device.data(); resource.res.pitch2D.desc=cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
        resource.res.pitch2D.width=width; resource.res.pitch2D.height=height; resource.res.pitch2D.pitchInBytes=width*4*sizeof(float);
        cudaTextureDesc settings{}; settings.filterMode=cudaFilterModeLinear; settings.normalizedCoords=true;
        settings.addressMode[0]=settings.addressMode[1]=settings.addressMode[2]=cudaAddressModeClamp;
        CUDA_CHECK_THROW(cudaCreateTextureObject(&texture,&resource,&settings,nullptr));
    }
    ~Image() { if (texture) cudaDestroyTextureObject(texture); }
};

struct Evaluation { json metrics; std::vector<float> prediction; std::vector<uint8_t> rgb; double inference_seconds; };
__global__ void quality_metrics(uint32_t n,const float* prediction,const float* linear,const uint8_t* reference,uint8_t* rgb,double* partial) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ double sums[4][256];
    double a=0,b=0,c=0,d=0;
    if (i<n) {
        float value=prediction[i];
        double error=(double)value-linear[(i/3)*4+i%3]; a=error*error;
        float display=powf(fminf(fmaxf(value,0.0f),1.0f),1.0f/2.2f);
        uint8_t byte=(uint8_t)(display*255.0f+0.5f);
        error=(double)display-reference[i]/255.0; b=error*error;
        error=((int)byte-(int)reference[i])/255.0; c=error*error;
        d=value<0 || value>1;
        if (rgb) rgb[i]=byte;
    }
    sums[0][threadIdx.x]=a; sums[1][threadIdx.x]=b; sums[2][threadIdx.x]=c; sums[3][threadIdx.x]=d;
    __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (threadIdx.x<(uint32_t)stride) for (int k=0;k<4;++k) sums[k][threadIdx.x]+=sums[k][threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) for (int k=0;k<4;++k) partial[blockIdx.x*4+k]=sums[k][0];
}

Evaluation evaluate(std::shared_ptr<NetworkWithInputEncoding<P>> network, cudaStream_t stream, const Image& image,bool capture=true) {
    GPUMatrix<float> output(3,image.padded);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); auto begin=Clock::now();
    network->inference(stream,GPUMatrix<float>{image.grid.data(),2,image.padded},output);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); double inference_time=seconds(begin);
    uint32_t blocks=(image.pixels*3+255)/256;
    GPUMemory<double> partial(blocks*4); GPUMemory<uint8_t> bytes(capture?image.pixels*3:0);
    quality_metrics<<<blocks,256,0,stream>>>(image.pixels*3,output.data(),image.device.data(),image.reference_device.data(),capture?bytes.data():nullptr,partial.data());
    CUDA_CHECK_THROW(cudaGetLastError()); CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    std::vector<double> host_partial(blocks*4); partial.copy_to_host(host_partial.data());
    double total[4]={0,0,0,0};
    for (uint32_t b=0;b<blocks;++b) for (int k=0;k<4;++k) total[k]+=host_partial[b*4+k];
    double count=image.pixels*3,linear_mse=total[0]/count,display_mse=total[1]/count,byte_mse=total[2]/count;
    if (!std::isfinite(linear_mse) || !std::isfinite(display_mse)) throw std::runtime_error("Non-finite decoded prediction");
    std::vector<float> prediction; std::vector<uint8_t> rgb;
    if (capture) {
        prediction.resize(image.pixels*3); rgb.resize(image.pixels*3);
        CUDA_CHECK_THROW(cudaMemcpy(prediction.data(),output.data(),prediction.size()*sizeof(float),cudaMemcpyDeviceToHost));
        bytes.copy_to_host(rgb.data());
    }
    auto psnr=[](double mse) { return -10.0*std::log10(std::max(mse,1e-30)); };
    return {json{{"linear_unclipped_mse",linear_mse},{"display_float_mse",display_mse},
        {"display_float_psnr_db",psnr(display_mse)},{"decoded8_mse",byte_mse},
        {"decoded8_psnr_db",psnr(byte_mse)},{"clipped_fraction",total[3]/count}},prediction,rgb,inference_time};
}

int main(int argc,char** argv) {
    try {
        if (argc!=5) throw std::runtime_error("Usage: bench_image_quality train|decode image config|model output");
        auto process_begin=Clock::now(); std::filesystem::path out=argv[4]; std::filesystem::create_directories(out);
        Image image(argv[2]); std::ifstream input(argv[3],std::ios::binary);
        if (!input) throw std::runtime_error("Missing config/model");
        bool decode=std::string(argv[1])=="decode";
        if (!decode && std::string(argv[1])!="train") throw std::runtime_error("Unknown mode");
        json packet,config;
        if (decode) {
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{}); packet=json::from_msgpack(bytes); config=packet.at("config");
            if (packet.at("width")!=image.width || packet.at("height")!=image.height) throw std::runtime_error("Image dimensions differ");
        } else config=json::parse(input);
        uint32_t seed=decode?1337:config.at("experiment").at("seed").get<uint32_t>();
        auto network=std::make_shared<NetworkWithInputEncoding<P>>(2,3,config.at("encoding"),config.at("network"));
        auto trainer=make_trainer(network,config,seed);
        cudaStream_t stream; CUDA_CHECK_THROW(cudaStreamCreate(&stream));
        if (decode) {
            auto load_begin=Clock::now(); trainer->deserialize(packet.at("model")); CUDA_CHECK_THROW(cudaDeviceSynchronize());
            double load_time=seconds(load_begin),setup=seconds(process_begin);
            evaluate(network,stream,image,false); std::vector<double> times; Evaluation result;
            for (int i=0;i<5;++i) { result=evaluate(network,stream,image,i==4); times.push_back(result.inference_seconds); }
            write_raw(out/"prediction.f32",result.prediction);
            if (!stbi_write_png((out/"reconstruction.png").string().c_str(),image.width,image.height,3,result.rgb.data(),image.width*3)) throw std::runtime_error("PNG write failed");
            write_json(out/"decode.json",json{{"status","pass"},{"metrics",result.metrics},{"width",image.width},{"height",image.height},
                {"load_setup_seconds",setup},{"parameter_restore_seconds",load_time},{"inference_seconds",times},
                {"model_bytes",std::filesystem::file_size(argv[3])}});
            CUDA_CHECK_THROW(cudaStreamDestroy(stream)); return 0;
        }
        auto experiment=config.at("experiment"); uint32_t batch=experiment.at("batch"),steps=experiment.at("steps"),interval=experiment.at("eval_every");
        if (batch!=1048576 || !steps || !interval) throw std::runtime_error("Expected million batch and positive step settings");
        // Canonical FullyFused initialization gives all backends identical flat FP32 weights.
        json canonical_opts=config.at("network"); canonical_opts["otype"]="FullyFusedMLP"; canonical_opts["sine_backward_mode"]="SavedZ";
        auto canonical=std::make_shared<NetworkWithInputEncoding<P>>(2,3,config.at("encoding"),canonical_opts);
        auto canonical_trainer=make_trainer(canonical,config,seed);
        if (canonical->n_params()!=network->n_params() || network->n_params()!=83968) throw std::runtime_error("Parameter layout/size mismatch");
        std::vector<float> initial(network->n_params());
        CUDA_CHECK_THROW(cudaMemcpy(initial.data(),canonical_trainer->params_full_precision(),initial.size()*sizeof(float),cudaMemcpyDeviceToHost));
        trainer->set_params_full_precision(initial.data(),initial.size()); write_raw(out/"initial-weights.f32",initial);
        canonical_trainer.reset(); canonical.reset(); json initial_state=trainer->serialize(true);
        GPUMemory<float> coordinates(batch*2); GPUMatrix<float> targets(3,batch); default_rng_t rng{seed};
        auto iteration=[&]() {
            generate_random_uniform<float>(stream,rng,batch*2,coordinates.data());
            linear_kernel(sample_pixels,0,stream,batch,(uint32_t)image.width,(uint32_t)image.height,image.texture,coordinates.data(),targets.data());
            return trainer->training_step(stream,GPUMatrix<float>{coordinates.data(),2,batch},targets);
        };
        auto warm_begin=Clock::now();
        for (uint32_t i=0;i<experiment.at("warmup").get<uint32_t>();++i) iteration();
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); double warmup_time=seconds(warm_begin);
        trainer->deserialize(initial_state); trainer->set_params_full_precision(initial.data(),initial.size()); rng=default_rng_t{seed};
        auto initial_eval_begin=Clock::now(); auto first=evaluate(network,stream,image);
        first.prediction.resize(std::min(first.prediction.size(),size_t{12288}));
        write_raw(out/"initial-prediction.f32",first.prediction);
        double initial_evaluation_time=seconds(initial_eval_begin);
        CUDA_CHECK_THROW(cudaDeviceSynchronize()); double preparation=seconds(process_begin);
        json curve=json::array(); double training_time=0,evaluation_time=0; Evaluation last;
        auto encode=[&]() {
            json decode_config={{"encoding",config.at("encoding")},{"network",config.at("network")}};
            decode_config["network"]["sine_backward_mode"]="SavedZ";
            json model={{"format","tcnn-image-fp16-v1"},{"width",image.width},{"height",image.height},
                {"transfer","stbi gamma2.2; clamp and round to uint8"},{"config",decode_config},{"model",trainer->serialize(false)}};
            return json::to_msgpack(model);
        };
        double best_psnr=-std::numeric_limits<double>::infinity(); json best_record;
        bool reached[4]={false,false,false,false};
        auto run_begin=Clock::now();
        for (uint32_t step=0;;) {
            auto eval_begin=Clock::now(); last=evaluate(network,stream,image,step==steps); evaluation_time+=seconds(eval_begin);
            json row=last.metrics; row["step"]=step; row["training_seconds"]=training_time; row["evaluation_seconds"]=evaluation_time; row["wall_seconds"]=seconds(run_begin);
            curve.push_back(row); write_json(out/"curve.json",curve);
            double psnr=row.at("decoded8_psnr_db");
            if (psnr>best_psnr) {
                best_psnr=psnr; best_record=row; write_raw(out/"best-model.msgpack",encode()); write_json(out/"best.json",row);
            }
            for (int k=0;k<4;++k) if (!reached[k] && psnr>=20+5*k) {
                reached[k]=true; write_raw(out/("target-"+std::to_string(20+5*k)+"db.msgpack"),encode());
            }
            std::cout<<"step="<<step<<" training_s="<<training_time<<" psnr8="<<row["decoded8_psnr_db"]<<std::endl;
            if (step==steps) break;
            uint32_t end=std::min(steps,step+interval); CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); auto begin=Clock::now();
            std::unique_ptr<TrainerT::ForwardContext> context;
            for (;step<end;++step) context=iteration();
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); training_time+=seconds(begin);
            float loss=trainer->loss(stream,*context); if (!std::isfinite(loss)) throw std::runtime_error("Non-finite training loss");
        }
        auto export_begin=Clock::now(); auto bytes=encode(); write_raw(out/"model.msgpack",bytes);
        double export_time=seconds(export_begin); write_raw(out/"prediction.f32",last.prediction);
        if (!stbi_write_png((out/"reconstruction.png").string().c_str(),image.width,image.height,3,last.rgb.data(),image.width*3)
            || !stbi_write_png((out/"reference.png").string().c_str(),image.width,image.height,3,image.reference.data(),image.width*3)) throw std::runtime_error("PNG write failed");
        write_json(out/"report.json",json{{"status","pass"},{"config",config},{"width",image.width},{"height",image.height},{"parameters",network->n_params()},
            {"training_seconds",training_time},{"evaluation_seconds",evaluation_time},{"warmup_seconds",warmup_time},{"preparation_seconds",preparation},
            {"model_export_seconds",export_time},{"initial_evaluation_seconds",initial_evaluation_time},
            {"encoding_seconds_estimate_setup_train_export",preparation-warmup_time-initial_evaluation_time+training_time+export_time},
            {"process_wall_seconds",seconds(process_begin)},{"model_bytes",bytes.size()},{"raw_fp16_parameter_bytes",network->n_params()*sizeof(P)},
            {"raw_rgb8_bytes",image.pixels*3},{"final",last.metrics},{"best",best_record},{"curve",curve}});
        CUDA_CHECK_THROW(cudaStreamDestroy(stream));
    } catch (const std::exception& error) { std::cerr<<error.what()<<std::endl; return 1; }
}
