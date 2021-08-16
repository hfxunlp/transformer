#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>
#include "../../../utils/cpp/base.h"

at::Tensor group_linear_forward(std::map<std::string, torch::Tensor> tensors, std::map<std::string, int64_t> iargs, std::map<std::string, bool> bargs) {

	auto inputu = tensors["inputu"];
	auto _sizes = inputu.sizes().vec();
	auto _s_last_dim_ind = _sizes.size() - 1;
	at::Tensor _id, out;
	auto trans_input = bargs["trans_input"];
	auto idm = inputu.dim();
	auto ngroup = iargs["ngroup"];
	if ((idm != 3) or trans_input) {
		int64_t _ldsize;
		if (trans_input) {
			_ldsize = iargs["isize"];
		}
		else {
			_ldsize = _sizes[_s_last_dim_ind];
		}
		_id = inputu.view({-1, ngroup, _ldsize});
	}
	else {
		_id = inputu;
	}
	_id = _id.transpose(0, 1);

	auto weight = tensors["weight"];
	auto bias = map_get(tensors, "bias");
	if (ct_is_none(bias)) {
		out = _id.bmm(weight);
	}
	else {
		out = bias.baddbmm(_id, weight);
	}
	if (bargs["shuffle"]) {
		out = out.permute({1, 2, 0});
	}
	else {
		out = out.transpose(0, 1);
	}

	_sizes[_s_last_dim_ind] = -1;
	if (bargs["i_gdim"]) {
		_sizes.insert(_sizes.end() - 1, ngroup);
	}
	else if (bargs["del_gdim"]) {
		_sizes.erase(_sizes.end() - 2);
	}

	return out.contiguous().view(_sizes);
}
