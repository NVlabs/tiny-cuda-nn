import torch
from tqdm import tqdm
import numpy as np
try:
    import tinycudann as tcnn
except:
    pass

class TcnnFCBlock(tcnn.Network):
    def __init__(
        self, in_features, out_features, 
        num_hidden_layers, hidden_features, 
        activation:str='ReLU', last_activation:str='None',
        seed=42):
        assert hidden_features in [16, 32, 64, 128], "hidden_features can only be 16, 32, 64, or 128."
        super().__init__(in_features, out_features, network_config={
            "otype": "FullyFusedMLP",               # Component type.
            "activation": activation,               # Activation of hidden layers.
            "output_activation": last_activation,   # Activation of the output layer.
            "n_neurons": hidden_features,           # Neurons in each hidden layer. # May only be 16, 32, 64, or 128.
            "n_hidden_layers": num_hidden_layers,   # Number of hidden layers.
            "feedback_alignment": False  # Use feedback alignment # [Lillicrap et al. 2016].
        }, seed=seed)
    
    def forward(self, x: torch.Tensor):
        prefix = x.shape[:-1]
        return super().forward(x.flatten(0,-2)).unflatten(0, prefix)

device = torch.device('cuda:0')
mlp = TcnnFCBlock(3, 256, 8, 128)

for _ in range(10000):
    for n, p in mlp.named_parameters():
        p.grad = None
    _x = np.random.randint(200, 1000, 1)[0]
    x = torch.rand([_x,1000,3], dtype=torch.float, device=device) # random setting
    #x = torch.rand([torch.randint(200,800,[1]).item(),100,3], dtype=torch.float, device=device) # setting 2
    y = mlp.forward(x)
    y.mean().backward()


# when under random setting line: 37
# now the total_n_bytes_allocated().load() can be stable, about 507510784(_x = np.random.randint(200, 800, 1)[0]
 #   x = torch.rand([_x,100,3], dtype=torch.float, device=device) # random setting)
# the previous version , the total_n_bytes_allocated().load() is continously increased, and there will be OOM risk, and we(with JianFei Guo) 
# are focusing on new encoding method, and we met this problem.

# BTW, and static std::atomic<size_t> s_total_n_bytes_allocated{0}; the size_t may can be change to ulong  or ullong to render more rays one time


# here is the mem allocation and free process of the previous verison,  the conclusion is that 
# when the input is random, the unreasonable free interval will cause more mem allocation. For example:
# in iter2  : free interval should adjusrt to [0, 23590400] [82112000, -331350016] not 
# [2572800, 2572800]   [0, 22182400]   [23590400, 23590400]    [22182400, 23590400]    [82112000, -331350016]

# the current version the total_n_bytes_allocated().load() can be stable

"""
iter0:

GPUMmeoryArena::allocate(): start=[       0], size=[2572800]                                                                                                                                                       
GPUMemoryArena::enlarge(): cnt[128] += [4194304]  
// m_allocated_intervals add  [0, 2572800]                                                                                                                                                           
Allocation: m_workspace=[272e7d70], m_offset=[       0], m_data=[ 2000000]                                                                                                                                         
~Allocation: m_workspace=[       0], cnt=[4194432]                                                                                                                                                                 
GPUMmeoryArena::allocate(): start=[  274200], size=[15436800]                                                                                                                                                      
GPUMemoryArena::enlarge(): cnt[4194432] += [14680064]                                                                                                                                                              
Allocation: m_workspace=[272e7d70], m_offset=[  274200], m_data=[ 2274200]  
// m_allocated_intervals add   [2572800, 18009600]                                                                                                                                   
~Allocation: m_workspace=[       0], cnt=[18874496]                                                                                                                                                                
~Allocation: m_workspace=[272e7d70], free(m_offset=[       0], m_data=[ 2000000]), cnt=[18874496]
// although free the memory (m_offset = 0), m_allocated_intervals erase [0, 2572800] 
// and the memory is still allocated, free interval add [0, 2572800]

*** m_allocated_intervals is: [2572800, 18009600]      [0, 2572800]                                                                                                                                               
 *** after erase m_allocated_intervals is: [2572800, 18009600]                                                                                                                                                     
 *** before merge free interval is: [0, 2572800]     [18009600, -331350016]                                                                                                                                             
 *** after merge free interval is:[0, 2572800]       [18009600, -331350016].

...

iter1:

//112ce00 -> 18009600
GPUMmeoryArena::allocate(): start=[ 112ce00], size=[5580800]
//5580800 < 2572800, allocate memory， m_allocated_intervals add [18009600, 23590400]
GPUMemoryArena::enlarge(): cnt[18874496] += [6291456]

~Allocation: m_workspace=[       0], cnt=[25165952]
GPUMmeoryArena::allocate(): start=[ 167f600], size=[33484800]
//33484800 < 2572800, allocate memory， m_allocated_intervals add [23590400, 57075200]
GPUMemoryArena::enlarge(): cnt[25165952] += [33554432] //25165952 + 33554432 = 58720384

~Allocation: m_workspace=[272e7d70], free(m_offset=[ 112ce00], m_data=[ 312ce00]), cnt=[58720384]
// although free the memory (m_offset = 272e7d70), m_allocated_intervals erase[18009600, 23590400]
// and the memory is still allocated, free interval add [18009600, 23590400]

*** m_allocated_intervals is: [23590400, 57075200]     [2572800, 18009600]     [18009600, 23590400] 
 *** after erase m_allocated_intervals is: [23590400, 57075200]         [2572800, 18009600] 
 *** before merge free interval is: [2572800, 2572800]       [0, 2572800]    [18009600, 23590400]    [57075200, -331350016] 
 *** after merge free interval is:[2572800, 2572800]         [0, 2572800]    [18009600, 23590400]    [57075200, -331350016]  GPUMmeoryArena::allocate(): start=[ 112ce00], size=[5580800]
Allocation: m_workspace=[272e7d70], m_offset=[ 112ce00], m_data=[ 312ce00]

iter2:
**********************
****** attention *****
***********************
GPUMmeoryArena::allocate(): start=[ 112ce00], size=[4172800] //don't enlarge
//5580800 > 4172800, we can use memory recorded in free interval [18009600, 23590400]，needn't allocate memory
// m_allocated_intervals add [18009600, 22182400], and free interval also add [22182400, 23590400]
//However, it is unreasonable, since the free memory is split into more chunks, and 
// it is hard to reuse again since every chunk may become so small 

Allocation: m_workspace=[272e7d70], m_offset=[ 112ce00], m_data=[ 312ce00]
~Allocation: m_workspace=[       0], cnt=[58720384]
GPUMmeoryArena::allocate(): start=[ 366e600], size=[25036800] //need 25036800

// 25036800 > free interval chunks，m_allocated_intervals add  [57075200, 82112000]
GPUMemoryArena::enlarge(): cnt[58720384] += [25165824]
Allocation: m_workspace=[272e7d70], m_offset=[ 366e600], m_data=[ 566e600]
~Allocation: m_workspace=[       0], cnt=[83886208]
~Allocation: m_workspace=[272e7d70], free(m_offset=[ 112ce00], m_data=[ 312ce00]), cnt=[83886208]

*** m_allocated_intervals is: [57075200, 82112000]     [18009600, 22182400]    [23590400, 57075200] 
 *** after erase m_allocated_intervals is: [57075200, 82112000]         [23590400, 57075200] //
 *** before merge free interval is: [2572800, 2572800]       [0, 18009600]   [18009600, 22182400]    [23590400, 23590400]    [22182400, 23590400]    [82112000, -331350016] 
 *** after merge free interval is:[2572800, 2572800]         [0, 22182400]   [23590400, 23590400]    [22182400, 23590400]    [82112000, -331350016]

**********************
****** attention *****
***********************
// free interval should adjust to [0, 23590400] [82112000, -331350016] not the last state
// [2572800, 2572800]         [0, 22182400]   [23590400, 23590400]    [22182400, 23590400]    [82112000, -331350016]


"""