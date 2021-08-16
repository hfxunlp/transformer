#ifndef _NEUTRON_MODULES_CPP_ACT_BASE
#define _NEUTRON_MODULES_CPP_ACT_BASE

#include <torch/torch.h>
#define _USE_MATH_DEFINES
#include <cmath>

typedef at::Tensor (*Act_Func) (torch::Tensor, bool inplace);

inline at::Tensor relu_forward(torch::Tensor x, bool inplace=false) {

	return torch::nn::functional::relu(x, torch::nn::functional::ReLUFuncOptions().inplace(inplace));
}

inline at::Tensor gelu_gpt_forward(torch::Tensor x, bool inplace=false) {

	return x * 0.5 * (1.0 + (sqrt(2.0 / M_PI) * (x + 0.044715 * x.pow(3.0))).tanh());
}

inline at::Tensor gelu_bert_forward(torch::Tensor x, bool inplace=false) {

	return x * 0.5 * (1.0 + (x / sqrt(2.0)).erf());
}

inline at::Tensor gelu_forward(torch::Tensor x, bool inplace=false) {

	return torch::nn::functional::gelu(x);
}

inline at::Tensor sigmoid_forward(torch::Tensor x, bool inplace=false) {

	return x.sigmoid();
}

inline at::Tensor swish_forward(torch::Tensor x, bool inplace=false) {

	return x.sigmoid() * x;
}

inline at::Tensor mish_forward(torch::Tensor x, bool inplace=false) {

	return x * torch::nn::functional::softplus(x).tanh();
}

#endif
