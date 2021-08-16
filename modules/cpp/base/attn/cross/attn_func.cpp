#define _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
#include "../common.cpp"
#undef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN

/*#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "../../../../../utils/cpp/base.h"

std::map<std::string, at::Tensor> attn_forward(std::map<std::string, torch::Tensor> tensors, std::map<std::string, int64_t> iargs, double p, double inf_value, std::map<std::string, bool> bargs) {

	auto iQ = tensors["iQ"];
	auto bsize = iQ.size(0);
	auto nquery = iQ.size(1);
	auto iK = tensors["iK"];
	auto seql = iK.size(1);
	auto nheads = iargs["num_head"];
	auto adim = iargs["attn_dim"];

	auto real_iQ = torch::nn::functional::linear(iQ, tensors["query_adaptor.weight"], map_get(tensors, "query_adaptor.bias")).view({bsize, nquery, nheads, adim}).transpose(1, 2);
	at::Tensor real_iK, real_iV;

	auto evaluation = not bargs["training"];
	auto buf_real_iK = map_get(tensors, "real_iK");
	auto buf_iK = map_get(tensors, "buf_iK");
	if (ct_is_not_none(buf_real_iK) and iK.is_set_to(buf_iK) and evaluation) {
		real_iK = buf_real_iK;
		real_iV = tensors["real_iV"];
	}
	else {
		auto _reals = torch::nn::functional::linear(iK, tensors["kv_adaptor.weight"], map_get(tensors, "kv_adaptor.bias")).view({bsize, seql, 2, nheads, adim}).unbind(2);
		real_iK = _reals[0].permute({0, 2, 3, 1});
		real_iV = _reals[1].transpose(1, 2);
	}

	auto scores = real_iQ.matmul(real_iK) / sqrt(adim);

	auto mask = map_get(tensors, "mask");
	if (ct_is_not_none(mask)) {
		scores.masked_fill_(mask.unsqueeze(1), -inf_value);
	}
	scores = scores.softmax(-1);
	if (p > 0.0) {
		scores = torch::nn::functional::dropout(scores, torch::nn::functional::DropoutFuncOptions().p(p).inplace(bargs["drop.inplace"]).training(bargs["drop.training"]));
	}

	auto out = torch::nn::functional::linear(scores.matmul(real_iV).transpose(1, 2).contiguous().view({bsize, nquery, nheads * adim}), tensors["outer.weight"], map_get(tensors, "outer.bias"));

	std::map<std::string, at::Tensor> rs;
	rs["out"] = out;
	if (evaluation) {
		rs["real_iK"] = real_iK;
		rs["real_iV"] = real_iV;
	}
	return rs;
}*/
