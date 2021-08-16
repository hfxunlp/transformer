#ifndef _NEUTRON_MODULES_CPP_GROUP_FUNC
#define _NEUTRON_MODULES_CPP_GROUP_FUNC

#include <torch/torch.h>
#include <map>
#include <string>

at::Tensor group_linear_forward(std::map<std::string, torch::Tensor> tensors, std::map<std::string, int64_t> iargs, std::map<std::string, bool> bargs);

#endif
