#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace wmma = nvcuda::wmma;

__global__ void probe(uint32_t* words, uint32_t* errors, int stride, int pattern) {
    __shared__ __half coordinates[16*136];
    for (int i=threadIdx.x+32*threadIdx.y;i<16*stride;i+=256)
        coordinates[i]=__float2half((float)((i/stride)*128+i%stride));
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
    wmma::fragment<wmma::accumulator,16,16,16,__half> c;
    static_assert(decltype(a)::num_elements>=decltype(c)::num_elements && decltype(c)::num_elements==8,"Unsupported consumed fragment cardinality");
    if (blockIdx.x==0 && threadIdx.x==0 && threadIdx.y==0) {
        errors[2]=a.num_elements; errors[3]=c.num_elements;
    }
    wmma::load_matrix_sync(a,coordinates+16*threadIdx.y,stride);
    wmma::load_matrix_sync(c,coordinates+16*threadIdx.y,stride,wmma::mem_row_major);
    const int base=1+64*blockIdx.x+8*threadIdx.y;
    for (int t=0;t<8;++t) {
        const bool flag=(((int)__half2float(c.x[t])*13+blockIdx.x*17+pattern*7)%19)<9;
        const uint32_t word=__ballot_sync(0xffffffffu,flag);
        if (threadIdx.x==0) words[base+t]=word;
    }
    __syncthreads();
    for (int t=0;t<8;++t) {
        const int id=(int)__half2float(a.x[t]);
        const bool expected=((id*13+blockIdx.x*17+pattern*7)%19)<9;
        if (a.x[t]!=c.x[t]) atomicAdd(errors,1u);
        if (((words[base+t]>>threadIdx.x)&1u)!=expected) atomicAdd(errors+1,1u);
    }
}

int main() {
    cudaDeviceProp actual;
    if (cudaGetDeviceProperties(&actual,0)!=cudaSuccess) return 1;
    const int arch=actual.major*10+actual.minor;
    if (arch!=80 && arch!=89 && arch!=120) { std::printf("SKIP unvalidated SignRecon architecture SM%d\n",arch); return 0; }
    constexpr uint32_t guard=0x5a71c39du;
    uint32_t *words=nullptr,*errors=nullptr;
    std::vector<uint32_t> host(32*64+2,guard);
    if (cudaMalloc(&words,host.size()*sizeof(uint32_t))!=cudaSuccess || cudaMalloc(&errors,4*sizeof(uint32_t))!=cudaSuccess) return 1;
    for (int stride:{128,136}) for (int pattern:{0,1,2}) {
        uint32_t counts[4]={0,0,0,0};
        if (cudaMemcpy(words,host.data(),host.size()*sizeof(uint32_t),cudaMemcpyHostToDevice)!=cudaSuccess ||
            cudaMemset(errors,0,sizeof(counts))!=cudaSuccess) return 2;
        probe<<<32,dim3(32,8)>>>(words,errors,stride,pattern);
        if (cudaGetLastError()!=cudaSuccess || cudaMemcpy(counts,errors,sizeof(counts),cudaMemcpyDeviceToHost)!=cudaSuccess ||
            cudaMemcpy(host.data(),words,host.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost)!=cudaSuccess) return 3;
        if (counts[0] || counts[1] || host.front()!=guard || host.back()!=guard) {
            std::printf("FAIL stride=%d pattern=%d fragment_mismatch=%u sign_mismatch=%u\n",stride,pattern,counts[0],counts[1]);
            return 4;
        }
        std::printf("PASS stride=%d pattern=%d:A_elements=%u,C_elements=%u;32 tiles,8 warps,32 lanes,8 consumed elements;guards intact\n",stride,pattern,counts[2],counts[3]);
    }
    cudaDeviceProp device;
    if (cudaGetDeviceProperties(&device,0)!=cudaSuccess) return 5;
    std::printf("PASS GPU=%s SM=%d%d compared_elements=393216\n",device.name,device.major,device.minor);
    cudaFree(errors);cudaFree(words);
}
