#include "test_common.h"
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

using namespace tcnn;
using P = network_precision_t;

namespace {
bool native_fused_available() { return MIN_GPU_ARCH>70 && TCNN_HALF_PRECISION; }
bool sign_supported() {
    const auto arch=cuda_compute_capability();
    return native_fused_available() && (arch==80 || arch==89 || arch==120);
}
template<class T> std::vector<T> cpu(const GPUMemory<T>& memory) {
    std::vector<T> result(memory.size()); memory.copy_to_host(result); return result;
}
void record(const json& row) {
    if (const char* path=std::getenv("TCNN_SINE_NUMERICS_REPORT")) {
        static json rows=json::array(); rows.push_back(row);
        std::ofstream stream(path); REQUIRE(stream.good()); stream<<rows.dump(2);
    }
}
template<class T> std::vector<float> floats(const std::vector<T>& values) {
    std::vector<float> result(values.size());
    std::transform(values.begin(),values.end(),result.begin(),[](T value){return (float)value;}); return result;
}
json difference(const std::vector<float>& actual,const std::vector<float>& reference) {
    REQUIRE(actual.size()==reference.size()); double error=0,a2=0,r2=0,dot=0,max_error=0,max_reference=0;
    std::vector<double> absolute; absolute.reserve(actual.size());
    bool finite=true;
    for (size_t i=0;i<actual.size();++i) {
        finite=finite && std::isfinite(actual[i]) && std::isfinite(reference[i]);
        double a=actual[i],r=reference[i],d=a-r; error+=d*d; a2+=a*a; r2+=r*r; dot+=a*r;
        absolute.push_back(std::abs(d)); max_error=std::max(max_error,std::abs(d)); max_reference=std::max(max_reference,std::abs(r));
    }
    REQUIRE(finite);
    std::sort(absolute.begin(),absolute.end());
    auto percentile=[&](double p){return absolute.empty()?0:absolute[(size_t)(p*(absolute.size()-1))];};
    return {{"max_abs",max_error},{"reference_max_abs",max_reference},{"error_norm",std::sqrt(error)},
        {"reference_norm",std::sqrt(r2)},{"relative_l2",r2>0?json(std::sqrt(error/r2)):json(nullptr)},
        {"cosine_similarity",a2>0 && r2>0?json(dot/std::sqrt(a2*r2)):json(nullptr)},
        {"p50_abs",percentile(.50)},{"p95_abs",percentile(.95)},{"p99_abs",percentile(.99)}};
}
void require_reference(const std::vector<float>& actual,const std::vector<float>& reference) {
    const auto metric=difference(actual,reference); INFO(metric.dump());
    // Sparse power-of-two fixtures isolate activation/state semantics from general GEMM rounding.
    if (metric["relative_l2"].is_null()) REQUIRE(metric["max_abs"].get<double>()==0);
    else REQUIRE(metric["relative_l2"].get<double>()<=0.01);
    REQUIRE(metric["max_abs"].get<double>()<=1e-4+0.005*metric["reference_max_abs"].get<double>());
}
__global__ void scalar_derivatives(uint32_t n,const __half* z,__half* h,__half* saved,__half* recon,float* native_cosine) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    h[i]=(__half)sinf((float)z[i]);
    saved[i]=activation_backward_in(Activation::Sine,(__half)1.0f,z[i]);
    native_cosine[i]=cosf((float)z[i]);
    recon[i]=sine_cosine_from_output(h[i],native_cosine[i]<0);
}
struct Result { std::vector<float> output,input_gradient,gradient; std::vector<std::vector<float>> hidden; };
struct Model {
    std::shared_ptr<Network<P>> network;
    GPUMemory<P> weights,gradients;
    Model(uint32_t input,uint32_t width,uint32_t depth,uint32_t output,bool recon,const std::vector<P>& params,bool output_sine=false)
        :weights(params.size()),gradients(params.size()) {
        json config={{"otype","FullyFusedMLP"},{"activation","Sine"},{"output_activation",output_sine?"Sine":"None"},
            {"n_input_dims",input},{"n_output_dims",output},{"n_neurons",width},{"n_hidden_layers",depth},
            {"sine_backward_mode",recon?"SignRecon":"SavedZ"}};
        network.reset(create_network<P>(config)); network->set_jit_fusion(false);
        REQUIRE(network->n_params()==params.size()); weights.copy_from_host(params.data());
        network->set_params(weights.data(),weights.data(),gradients.data());
    }
    Result run(const std::vector<P>& input_values,const std::vector<P>& upstream,uint32_t batch,bool with_input=true,bool capture_hidden=true) {
        GPUMatrix<P> input(network->input_width(),batch),output(network->padded_output_width(),batch);
        GPUMatrix<P> dy(network->padded_output_width(),batch),dx(network->input_width(),with_input?batch:0);
        CUDA_CHECK_THROW(cudaMemcpy(input.data(),input_values.data(),input_values.size()*sizeof(P),cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(dy.data(),upstream.data(),upstream.size()*sizeof(P),cudaMemcpyHostToDevice));
        auto ctx=network->forward(input,&output,false,with_input);
        network->backward(*ctx,input,output,dy,with_input?&dx:nullptr);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        Result result{floats(output.to_cpu_vector()),with_input?floats(dx.to_cpu_vector()):std::vector<float>{},floats(cpu(gradients)),{}};
        for (uint32_t l=0;capture_hidden && l<network->num_forward_activations();++l) {
            const P* pointer=network->forward_activations(*ctx,l).first;
            std::vector<P> values(network->width(l)*batch);
            CUDA_CHECK_THROW(cudaMemcpy(values.data(),pointer,values.size()*sizeof(P),cudaMemcpyDeviceToHost));
            result.hidden.push_back(floats(values));
        }
        return result;
    }
};
struct Fixture {
    uint32_t input,width,depth,output,batch,padded_output;
    bool output_sine;
    std::vector<P> weights,x,dy;
    std::vector<size_t> offsets;
    Fixture(uint32_t i,uint32_t w,uint32_t d,uint32_t o,uint32_t b,uint32_t projection=0,bool sine_output=false)
        :input(i),width(w),depth(d),output(o),batch(b),padded_output(next_multiple(o,16u)),output_sine(sine_output) {
        size_t total=0;
        for (uint32_t l=0;l<=depth;++l) { offsets.push_back(total); total+=(l==depth?padded_output:width)*(l==0?input:width); }
        weights.assign(total,(P)0);
        for (uint32_t l=0;l<depth;++l) for (uint32_t row=0;row<width;++row) {
            uint32_t cols=l==0?input:width, col=(row*5+l*7)%cols;
            weights[offsets[l]+row*cols+col]=(P)((row%2?-1:1)*(l==0?1:2));
        }
        for (uint32_t row=0;row<output;++row) weights[offsets[depth]+row*width+(row+projection)%width]=(P)(row%2?-1:1);
        const float states[]={0,0.00004f,-0.00004f,.25f,-.25f,.5f,-.5f,1,-1,1.5703125f,1.5712890625f,
            -1.5703125f,-1.5712890625f,3.140625f,-3.140625f,4.7109375f,-4.7109375f,8,-8,31.5f,-31.5f};
        x.resize(batch*input); dy.assign(batch*padded_output,(P)0);
        for (size_t j=0;j<x.size();++j) x[j]=(P)states[(j*13+j/input*7)%(sizeof(states)/sizeof(states[0]))];
        for (uint32_t p=0;p<batch;++p) for (uint32_t row=0;row<output;++row) dy[p*padded_output+row]=(P)(((int)((p*3+row*5)%7)-3)/2048.0f);
    }
    Result reference(bool recon) const {
        Result result; std::vector<std::vector<float>> h{floats(x)},z;
        auto round=[](double value){return (float)(P)(float)value;};
        for (uint32_t l=0;l<=depth;++l) {
            uint32_t rows=l==depth?padded_output:width,cols=l==0?input:width;
            std::vector<float> next(batch*rows),pre(batch*rows);
            for (uint32_t p=0;p<batch;++p) for (uint32_t r=0;r<rows;++r) {
                double sum=0; for (uint32_t c=0;c<cols;++c) sum+=(float)weights[offsets[l]+r*cols+c]*h.back()[p*cols+c];
                pre[p*rows+r]=round(sum); next[p*rows+r]=(l<depth || output_sine)?round(std::sin((double)pre[p*rows+r])):pre[p*rows+r];
            }
            z.push_back(pre); h.push_back(next);
        }
        result.output=h.back(); result.hidden.assign(h.begin()+1,h.end()-1); result.gradient.resize(weights.size());
        std::vector<float> delta=floats(dy);
        if (output_sine) for (size_t j=0;j<delta.size();++j) delta[j]=round(delta[j]*round(std::cos((double)z.back()[j])));
        for (int l=(int)depth;l>=0;--l) {
            uint32_t rows=l==(int)depth?padded_output:width,cols=l==0?input:width;
            for (uint32_t r=0;r<rows;++r) for (uint32_t c=0;c<cols;++c) {
                double sum=0; for (uint32_t p=0;p<batch;++p) sum+=delta[p*rows+r]*h[l][p*cols+c];
                result.gradient[offsets[l]+r*cols+c]=round(sum);
            }
            std::vector<float> previous(batch*cols);
            for (uint32_t p=0;p<batch;++p) for (uint32_t c=0;c<cols;++c) {
                double sum=0; for (uint32_t r=0;r<rows;++r) sum+=(float)weights[offsets[l]+r*cols+c]*delta[p*rows+r];
                float grad=round(sum);
                if (l>0) {
                    double derivative=std::cos((double)z[l-1][p*cols+c]);
                    if (recon) {
                        double value=h[l][p*cols+c]; double magnitude=std::sqrt(std::max(0.0,1-value*value));
                        derivative=derivative<0?-magnitude:magnitude;
                    }
                    grad=round(grad*round(derivative));
                }
                previous[p*cols+c]=grad;
            }
            delta=std::move(previous);
        }
        result.input_gradient=delta; return result;
    }
};
void check_fixture(const Fixture& f,bool recon) {
    Model model(f.input,f.width,f.depth,f.output,recon,f.weights,f.output_sine);
    auto actual=model.run(f.x,f.dy,f.batch); auto reference=f.reference(recon);
    require_reference(actual.output,reference.output); require_reference(actual.input_gradient,reference.input_gradient);
    require_reference(actual.gradient,reference.gradient);
    for (size_t l=0;l<actual.hidden.size();++l) require_reference(actual.hidden[l],reference.hidden[l]);
    GPUMatrix<P> output(model.network->padded_output_width(),f.batch),input(f.input,f.batch);
    CUDA_CHECK_THROW(cudaMemcpy(input.data(),f.x.data(),f.x.size()*sizeof(P),cudaMemcpyHostToDevice));
    model.network->inference_mixed_precision(input,output,false);
    require_reference(floats(output.to_cpu_vector()),reference.output);
    record({{"kind","controlled_network"},{"mode",recon?"SignRecon":"SavedZ"},{"input",f.input},{"width",f.width},
        {"depth",f.depth},{"output",f.output},{"batch",f.batch},{"gradient_reference",difference(actual.gradient,reference.gradient)},
        {"input_gradient_reference",difference(actual.input_gradient,reference.input_gradient)}});
}
}

TEST_CASE("Sine scalar derivatives obey their declared FP16 semantics","[sine][numerics]") {
    tcnn_test_setup(); std::vector<__half> values;
    for (uint32_t bits=0;bits<65536;++bits) if ((bits&0x7c00)!=0x7c00) {
        __half_raw raw; raw.x=(uint16_t)bits; values.push_back(__half(raw));
    }
    GPUMemory<__half> z(values.size()),h(values.size()),saved(values.size()),recon(values.size()); z.copy_from_host(values.data());
    GPUMemory<float> native_cosine(values.size());
    scalar_derivatives<<<(values.size()+255)/256,256>>>(values.size(),z.data(),h.data(),saved.data(),recon.data(),native_cosine.data());
    CUDA_CHECK_THROW(cudaGetLastError()); auto hs=cpu(h),a=cpu(saved),b=cpu(recon);
    auto native=cpu(native_cosine);
    std::vector<float> ideal,af=floats(a),bf=floats(b); uint32_t lost=0,saturated=0;
    for (size_t i=0;i<values.size();++i) {
        double c=std::cos((double)(float)values[i]),v=(float)hs[i]; ideal.push_back((float)c);
        double magnitude=std::sqrt(std::max(0.0,1-v*v)); float expected=(float)(__half)(float)(native[i]<0?-magnitude:magnitude);
        REQUIRE(af[i]==(float)(__half)native[i]);
        // Fast-math cosine has a documented bounded-argument accuracy contract.
        // Retain the original threshold in [-pi,pi]; report all larger angles without a uniform accuracy gate.
#ifdef TCNN_RTC_USE_FAST_MATH
        if (std::abs((float)values[i])<=3.141592653589793) REQUIRE(std::abs(af[i]-c)<=5e-4);
#else
        REQUIRE(std::abs(af[i]-c)<=5e-4);
#endif
        REQUIRE(std::abs(bf[i]-expected)<=5e-4);
        REQUIRE(std::isfinite(bf[i])); saturated+=std::abs(v)==1; lost+=bf[i]==0 && af[i]!=0;
        if (bf[i]!=0) REQUIRE(std::signbit(bf[i])==(native[i]<0));
    }
    REQUIRE(lost>0); REQUIRE(saturated>0);
    record({{"kind","scalar_all_finite_half_angles"},{"elements",values.size()},{"saturated",saturated},
        {"nonzero_saved_cosine_reconstructed_as_zero",lost},{"SavedZ_vs_FP64",difference(af,ideal)},
        {"SignRecon_vs_FP64",difference(bf,ideal)},{"SignRecon_vs_SavedZ",difference(bf,af)}});
}

TEST_CASE("SavedZ Sine matches an independent rounded dense reference","[sine][reference]") {
    tcnn_test_setup(); if (!native_fused_available()) { WARN("Native FP16 FullyFused unavailable in this build"); return; }
    auto [input,width,depth,output,batch]=GENERATE(table<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>({
        {16,16,1,1,256},{32,32,2,3,512},{64,64,3,16,256},{128,128,5,16,512},{16,32,2,17,256}}));
    CAPTURE(input,width,depth,output,batch); check_fixture(Fixture(input,width,depth,output,batch),false);
}

TEST_CASE("Sine output activation preserves its own pre-activation","[sine][reference][output]") {
    tcnn_test_setup(); if (!native_fused_available()) return;
    uint32_t output=GENERATE(3u,17u);
    Fixture fixture(16,32,2,output,256,0,true); check_fixture(fixture,false);
    Model model(16,32,2,output,false,fixture.weights,true);
    GPUMatrix<P> input(16,256),fake_output(fixture.padded_output,256),dy(fixture.padded_output,256),dx(16,256);
    CUDA_CHECK_THROW(cudaMemcpy(input.data(),fixture.x.data(),fixture.x.size()*sizeof(P),cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(dy.data(),fixture.dy.data(),fixture.dy.size()*sizeof(P),cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMemset(fake_output.data(),0,fake_output.n_bytes()));
    auto context=model.network->forward(input,nullptr,false,true);
    model.network->backward(*context,input,fake_output,dy,&dx);
    require_reference(floats(cpu(model.gradients)),fixture.reference(false).gradient);
}

TEST_CASE("SavedZ Sine output supports row-major input and output","[sine][reference][layout]") {
    tcnn_test_setup(); if (!native_fused_available()) return;
    Fixture fixture(16,32,2,3,256,0,true); Model model(16,32,2,3,false,fixture.weights,true);
    auto to_rm=[](const std::vector<P>& values,uint32_t rows,uint32_t batch) {
        std::vector<P> result(values.size());
        for (uint32_t p=0;p<batch;++p) for (uint32_t c=0;c<rows;++c) result[c*batch+p]=values[p*rows+c];
        return result;
    };
    auto to_cm=[](const std::vector<P>& values,uint32_t rows,uint32_t batch) {
        std::vector<float> result(values.size());
        for (uint32_t p=0;p<batch;++p) for (uint32_t c=0;c<rows;++c) result[p*rows+c]=(float)values[c*batch+p];
        return result;
    };
    GPUMatrixDynamic<P> input(16,256,RM),output(16,256,RM),dy(16,256,RM),dx(16,256,RM);
    auto x=to_rm(fixture.x,16,256),upstream=to_rm(fixture.dy,16,256);
    CUDA_CHECK_THROW(cudaMemcpy(input.data(),x.data(),input.n_bytes(),cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(dy.data(),upstream.data(),dy.n_bytes(),cudaMemcpyHostToDevice));
    auto context=model.network->forward(input,&output,false,true);
    model.network->backward(*context,input,output,dy,&dx);
    const auto reference=fixture.reference(false);
    require_reference(to_cm(output.to_cpu_vector(),16,256),reference.output);
    require_reference(to_cm(dx.to_cpu_vector(),16,256),reference.input_gradient);
    require_reference(floats(cpu(model.gradients)),reference.gradient);
    model.network->inference_mixed_precision(input,output,false);
    require_reference(to_cm(output.to_cpu_vector(),16,256),reference.output);
}

#if TCNN_MIN_GPU_ARCH > 70 && TCNN_HALF_PRECISION
TEST_CASE("SignRecon Sine matches its independent reconstruction reference","[sine][signrecon][reference]") {
    tcnn_test_setup(); if (!sign_supported()) { WARN("SignRecon unavailable on this GPU"); return; }
    uint32_t depth=GENERATE(1u,2u,5u);
    for (uint32_t projection=0;projection<128;projection+=16) {
        CAPTURE(depth,projection); Fixture fixture(128,128,depth,16,256,projection); check_fixture(fixture,true);
        Model saved(128,128,depth,16,false,fixture.weights),recon(128,128,depth,16,true,fixture.weights);
        auto a=saved.run(fixture.x,fixture.dy,256),b=recon.run(fixture.x,fixture.dy,256);
        REQUIRE(a.output==b.output); REQUIRE(a.hidden==b.hidden);
        record({{"kind","controlled_mode_difference"},{"depth",depth},{"projection",projection},
            {"parameters",difference(b.gradient,a.gradient)},{"input",difference(b.input_gradient,a.input_gradient)}});
    }
}

TEST_CASE("SignRecon invalid configurations are rejected","[sine][signrecon]") {
    tcnn_test_setup(); json config={{"otype","FullyFusedMLP"},{"activation","Sine"},{"output_activation","None"},
        {"n_input_dims",128},{"n_output_dims",3},{"n_neurons",128},{"n_hidden_layers",5},{"sine_backward_mode","SignRecon"}};
    for (const json& update : std::vector<json>{{{"n_neurons",64}},{{"n_input_dims",16}},{{"n_output_dims",17}},{{"output_activation","Sine"}},{{"activation","ReLU"}}}) {
        auto changed=config; changed.update(update); REQUIRE_THROWS_AS(std::unique_ptr<Network<P>>(create_network<P>(changed)),std::runtime_error);
    }
    if (!sign_supported()) {
        REQUIRE_THROWS_AS(std::unique_ptr<Network<P>>(create_network<P>(config)),std::runtime_error);
        return;
    }
    Fixture fixture(128,128,1,3,256); Model model(128,128,1,3,true,fixture.weights);
    REQUIRE(model.network->hyperparams()["sine_backward_mode"]=="SignRecon");
    REQUIRE_THROWS_AS(model.network->generate_device_function("unsupported_jit"),std::runtime_error);
    GPUMatrixDynamic<P> input(128,256,RM); GPUMatrix<P> output(16,256);
    REQUIRE_THROWS_AS(model.network->forward(input,&output,false,false),std::runtime_error);
    auto invalid=config; invalid["sine_backward_mode"]="unknown";
    REQUIRE_THROWS_AS(std::unique_ptr<Network<P>>(create_network<P>(invalid)),std::runtime_error);
}

TEST_CASE("Compare Sine modes on image checkpoints without an equivalence gate","[sine][.measurement]") {
    tcnn_test_setup(); REQUIRE(sign_supported()); const char* directory=std::getenv("TCNN_SINE_QUALITY_ROOT");
    REQUIRE(directory!=nullptr); std::filesystem::path root(directory);
    std::ifstream report_file(root/"reports.json"); REQUIRE(report_file.good());
    json reports=json::parse(report_file); std::set<uint32_t> seeds;
    for (const auto& report:reports) seeds.insert(report.at("seed").get<uint32_t>());
    REQUIRE(!seeds.empty());
    auto feature_values=[](uint32_t batch) {
        GPUMatrix<float> coordinates(2,batch); std::vector<float> host(batch*2);
        uint32_t state=1337;
        for (float& value:host) { state=state*1664525u+1013904223u; value=((state>>8)+.5f)/16777216.0f; }
        CUDA_CHECK_THROW(cudaMemcpy(coordinates.data(),host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice));
        std::shared_ptr<Encoding<P>> encoding{create_encoding<P>(2,{{"otype","OneBlob"},{"n_bins",64}},16)};
        GPUMatrix<P> features(128,batch); encoding->inference_mixed_precision(coordinates,features);
        return features.to_cpu_vector();
    };
    float upstream_scale=1.28f;
    if (const char* scale=std::getenv("TCNN_SINE_VJP_SCALE")) {
        char* end=nullptr; upstream_scale=std::strtof(scale,&end);
        REQUIRE(end!=scale); REQUIRE(*end=='\0'); REQUIRE(std::isfinite(upstream_scale)); REQUIRE(upstream_scale>0);
    }
    std::vector<uint32_t> batches{256,4096};
    if (std::getenv("TCNN_SINE_LARGE_ONLY")) batches={1048576};
    else if (std::getenv("TCNN_SINE_LARGE_MEASUREMENT")) batches.push_back(1048576);
    for (uint32_t batch:batches) {
        auto features=feature_values(batch);
        std::vector<P> upstream(batch*16,(P)0);
        for (uint32_t p=0;p<batch;++p) for (uint32_t c=0;c<3;++c)
            upstream[p*16+c]=(P)(((int)((p*17+c*23)%11)-5)*upstream_scale/batch);
        for (uint32_t seed:seeds) {
            std::vector<std::pair<std::string,std::filesystem::path>> checkpoints;
            if (batch<1048576) {
                checkpoints.emplace_back("initial",root/("seed"+std::to_string(seed))/"saved-z"/"initial-weights.f32");
                for (const std::string origin:{"saved-z","fragment-sign"}) for (const std::string snapshot:{"target-25db","best-model","model"})
                    checkpoints.emplace_back(origin+"/"+snapshot,root/("seed"+std::to_string(seed))/origin/(snapshot+".msgpack"));
            } else checkpoints.emplace_back("saved-z/best-model",root/("seed"+std::to_string(seed))/"saved-z"/"best-model.msgpack");
            for (const auto& [name,path]:checkpoints) {
                CAPTURE(seed,batch,name); std::ifstream file(path,std::ios::binary); REQUIRE(file.good());
                std::vector<P> weights(83968);
                if (name=="initial") {
                    std::vector<float> values(weights.size()); file.read((char*)values.data(),values.size()*sizeof(float)); REQUIRE(file.good());
                    std::transform(values.begin(),values.end(),weights.begin(),[](float value){return (P)value;});
                } else {
                    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),{}); auto packet=json::from_msgpack(bytes);
                    REQUIRE(packet["format"]=="tcnn-image-fp16-v1"); REQUIRE(packet["model"]["params_type"]=="__half");
                    REQUIRE(packet["config"]["encoding"]["otype"]=="OneBlob"); REQUIRE(packet["config"]["encoding"]["n_bins"]==64);
                    REQUIRE(packet["config"]["network"]["n_neurons"]==128); REQUIRE(packet["config"]["network"]["n_hidden_layers"]==5);
                    const auto& data=packet["model"]["params_binary"].get_binary(); REQUIRE(data.size()==weights.size()*sizeof(P));
                    std::memcpy(weights.data(),data.data(),data.size());
                }
                Model saved(128,128,5,3,false,weights),recon(128,128,5,3,true,weights);
                bool with_input=batch<1048576;
                auto a=saved.run(features,upstream,batch,with_input,false),b=recon.run(features,upstream,batch,with_input,false);
                std::vector<float> rgb_a,rgb_b; rgb_a.reserve(batch*3); rgb_b.reserve(batch*3);
                for (uint32_t p=0;p<batch;++p) for (uint32_t c=0;c<3;++c) {rgb_a.push_back(a.output[p*16+c]); rgb_b.push_back(b.output[p*16+c]);}
                // This gate only checks forward invariance; backward differences are reported, not required to vanish.
                require_reference(rgb_b,rgb_a);
                json layers=json::array(); size_t offset=0;
                for (uint32_t l=0;l<6;++l) {
                    size_t count=l<5?128*128:16*128;
                    layers.push_back({{"layer",l},{"difference",difference(
                        std::vector<float>(b.gradient.begin()+offset,b.gradient.begin()+offset+count),
                        std::vector<float>(a.gradient.begin()+offset,a.gradient.begin()+offset+count))}}); offset+=count;
                }
                record({{"kind","image_checkpoint"},{"seed",seed},{"batch",batch},{"checkpoint",name},
                    {"source",path.string()},{"reference","SavedZ at identical FP16 weights and upstream"},
                    {"upstream","Deterministic signed VJP; scale/batch; not optimizer or RelativeL2 trajectory"},
                    {"upstream_scale",upstream_scale},
                    {"forward",difference(rgb_b,rgb_a)},{"parameters",difference(b.gradient,a.gradient)},{"layers",layers},
                    {"input_features",with_input?difference(b.input_gradient,a.input_gradient):json(nullptr)}});
            }
        }
    }
}
#endif

TEST_CASE("Sine zero gradients and live contexts preserve gradient modes","[sine][context]") {
    tcnn_test_setup(); if (!native_fused_available()) { WARN("Native FP16 FullyFused unavailable in this build"); return; }
    std::vector<bool> modes{false};
#if TCNN_MIN_GPU_ARCH > 70 && TCNN_HALF_PRECISION
    if (sign_supported()) modes.push_back(true);
#endif
    for (bool recon:modes) {
        CAPTURE(recon); Fixture f(128,128,5,3,256); Model model(128,128,5,3,recon,f.weights);
        GPUMatrix<P> input(128,256),out_a(16,256),out_b(16,256),dy(16,256),dx(128,256);
        CUDA_CHECK_THROW(cudaMemcpy(input.data(),f.x.data(),f.x.size()*sizeof(P),cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(dy.data(),f.dy.data(),f.dy.size()*sizeof(P),cudaMemcpyHostToDevice));
        auto independent_a=model.run(f.x,f.dy,256);
        auto other_x=f.x; for (size_t i=0;i<other_x.size();++i) other_x[i]=f.x[(i+13)%f.x.size()];
        auto independent_b=model.run(other_x,f.dy,256);
        auto a=model.network->forward(input,&out_a,false,true);
        GPUMatrix<P> other_input(128,256);
        CUDA_CHECK_THROW(cudaMemcpy(other_input.data(),other_x.data(),other_x.size()*sizeof(P),cudaMemcpyHostToDevice));
        auto b=model.network->forward(other_input,&out_b,false,true);
        model.network->backward(*b,other_input,out_b,dy,&dx); auto first=cpu(model.gradients);
        require_reference(floats(first),independent_b.gradient);
        model.network->backward(*a,input,out_a,dy,&dx,false,GradientMode::Accumulate);
        auto accumulated=floats(cpu(model.gradients)),expected=independent_a.gradient;
        for (size_t i=0;i<expected.size();++i) expected[i]=(float)(P)(expected[i]+independent_b.gradient[i]);
        require_reference(accumulated,expected);
        model.network->backward(*b,other_input,out_b,dy,&dx,false,GradientMode::Ignore);
        REQUIRE(floats(cpu(model.gradients))==accumulated);
        CUDA_CHECK_THROW(cudaMemset(dy.data(),0,dy.n_bytes()));
        model.network->backward(*a,input,out_a,dy,&dx,false,GradientMode::Overwrite);
        auto zero=cpu(model.gradients); auto zero_dx=dx.to_cpu_vector();
        REQUIRE(std::all_of(zero.begin(),zero.end(),[](P v){return (float)v==0;}));
        REQUIRE(std::all_of(zero_dx.begin(),zero_dx.end(),[](P v){return (float)v==0;}));
    }
}
