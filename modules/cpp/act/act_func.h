#ifndef _NEUTRON_MODULES_CPP_ACT_ACT_FUNC
#define _NEUTRON_MODULES_CPP_ACT_FUNC

#include <torch/torch.h>
#include <string>
#include "base.h"

const void* get_func(std::string func_name);

at::Tensor act_forward(void* act, torch::Tensor input, bool inplace=false);

#endif
