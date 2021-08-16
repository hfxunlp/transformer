#include <string>
#include <torch/torch.h>
#include "base.h"

inline Act_Func get_func_core(std::string func_name) {
	if (func_name == "gelu") {
		return gelu_forward;
	}
	else if (func_name == "swish") {
		return swish_forward;
	}
	else if (func_name == "sigmoid") {
		return sigmoid_forward;
	}
	else if (func_name == "mish") {
		return mish_forward;
	}
	else {
		return relu_forward;
	}
}

const void* get_func(std::string func_name) {
	return (void*)get_func_core(func_name);
}

at::Tensor act_forward(void* act, torch::Tensor input, bool inplace=false) {
	return (*(Act_Func)act)(input, inplace=inplace);
}
