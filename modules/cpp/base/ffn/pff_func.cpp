#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>
#include "../../act/act_func.h"
#include "../../../../utils/cpp/base.h"

at::Tensor positionwise_ff_forward(std::map<std::string, torch::Tensor> tensors, std::map<std::string, bool> bargs, void* act, std::map<std::string, double> dargs, std::vector<int64_t> normalized_shape) {

	auto x = tensors["x"];

	auto ln_opts = torch::nn::functional::LayerNormFuncOptions(normalized_shape).eps(dargs["normer.eps"]);
	auto ln_weight = map_get(tensors, "normer.weight");
	if (ct_is_not_none(ln_weight)) {
		ln_opts = ln_opts.weight(ln_weight);
	}
	auto ln_bias = map_get(tensors, "normer.bias");
	if (ct_is_not_none(ln_bias)) {
		ln_opts = ln_opts.bias(ln_bias);
	}

	auto _out = torch::nn::functional::layer_norm(x, ln_opts);

	auto p = dargs["net.2.p"];

	auto out = torch::nn::functional::linear(_out, tensors["net.0.weight"], map_get(tensors, "net.0.bias"));
	out = act_forward(act, out, bargs["net.1.inplace"]);
	if (p > 0.0) {
		out = torch::nn::functional::dropout(out, torch::nn::functional::DropoutFuncOptions().p(p).training(bargs["net.2.training"]).inplace(bargs["net.2.inplace"]));
		out = torch::nn::functional::linear(out, tensors["net.3.weight"], map_get(tensors, "net.3.bias"));
		out = torch::nn::functional::dropout(out, torch::nn::functional::DropoutFuncOptions().p(p).training(bargs["net.4.training"]).inplace(bargs["net.4.inplace"]));
	}
	else {
		out = torch::nn::functional::linear(out, tensors["net.2.weight"], map_get(tensors, "net.2.bias"));
	}

	if (bargs["norm_residual"]) {
		out = out + _out;
	}
	else {
		out = out + x;
	}

	return out;
}
