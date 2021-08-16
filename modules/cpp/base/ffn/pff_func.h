#ifndef _NEUTRON_MODULES_CPP_BASE_FFN_PFF_FUNC
#define _NEUTRON_MODULES_CPP_BASE_FFN_PFF_FUNC

#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>

at::Tensor positionwise_ff_forward(std::map<std::string, torch::Tensor> tensors, std::map<std::string, bool> bargs, void* act, std::map<std::string, double> dargs, std::vector<int64_t> normalized_shape);

#endif
