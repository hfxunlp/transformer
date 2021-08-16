#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "../../../../utils/cpp/base.h"

#if !(defined(_NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN)||defined(_NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN)||defined(_NEUTRON_MODULES_BASE_ATTN_BUILD_MHATTN))
#define _NEUTRON_MODULES_BASE_ATTN_BUILD_MHATTN
#endif

#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
#define _NEUTRON_MODULES_BASE_ATTN_FUNC_NAME self_attn_forward
#elif defined(_NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN)
#define _NEUTRON_MODULES_BASE_ATTN_FUNC_NAME cross_attn_forward
#else
#define _NEUTRON_MODULES_BASE_ATTN_FUNC_NAME multi_head_attn_forward
#endif

#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
#include "base.h"
std::map<std::string, at::Tensor> _NEUTRON_MODULES_BASE_ATTN_FUNC_NAME(std::map<std::string, torch::Tensor> tensors, std::vector<torch::Tensor> states, std::map<std::string, int64_t> iargs, double p, double inf_value, std::map<std::string, bool> bargs) {
#else
std::map<std::string, at::Tensor> _NEUTRON_MODULES_BASE_ATTN_FUNC_NAME(std::map<std::string, torch::Tensor> tensors, std::map<std::string, int64_t> iargs, double p, double inf_value, std::map<std::string, bool> bargs) {
#endif

	auto iQ = tensors["iQ"];
	auto bsize = iQ.size(0);
	auto nquery = iQ.size(1);
	#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
	auto iK = tensors["iK"];
	auto seql = iK.size(1);
	#endif
	#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_MHATTN
	auto iV = tensors["iV"];
	#endif
	auto nheads = iargs["num_head"];
	auto adim = iargs["attn_dim"];
	#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
	int64_t seql;
	#endif
	#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
	torch::Tensor rel_pos_cache;
	#endif

	#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
	auto real_iQ = torch::nn::functional::linear(iQ, tensors["query_adaptor.weight"], map_get(tensors, "query_adaptor.bias")).view({bsize, nquery, nheads, adim}).transpose(1, 2);
	at::Tensor real_iK, real_iV;

	auto evaluation = not bargs["training"];
	auto buf_real_iK = map_get(tensors, "real_iK");
	auto buf_iK = map_get(tensors, "buf_iK");
	if (ct_is_not_none(buf_real_iK) and iK.is_set_to(buf_iK) and evaluation) {
		real_iK = buf_real_iK;
		#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
		real_iV = tensors["real_iV"];
		#endif
	}
	else {
		#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
		auto _reals = torch::nn::functional::linear(iK, tensors["kv_adaptor.weight"], map_get(tensors, "kv_adaptor.bias")).view({bsize, seql, 2, nheads, adim}).unbind(2);
		real_iK = _reals[0].permute({0, 2, 3, 1});
		real_iV = _reals[1].transpose(1, 2);
		#else
		real_iK = torch::nn::functional::linear(iK, tensors["key_adaptor.weight"], map_get(tensors, "key_adaptor.bias")).view({bsize, seql, nheads, adim}).permute({0, 2, 3, 1});
		#endif
	}
	#else
	auto _reals = torch::nn::functional::linear(iQ, tensors["adaptor.weight"], map_get(tensors, "adaptor.bias")).view({bsize, nquery, 3, nheads, adim}).unbind(2);
	auto real_iQ = _reals[0].transpose(1, 2);
	auto real_iK = _reals[1].permute({0, 2, 3, 1});
	auto real_iV = _reals[2].transpose(1, 2);
	#endif

	#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_MHATTN
	auto buf_real_iV = map_get(tensors, "real_iV");
	auto buf_iV = map_get(tensors, "buf_iV");
	if (ct_is_not_none(buf_real_iV) and iV.is_set_to(buf_iV) and evaluation) {
		real_iV = buf_real_iV;
	}
	else {
		real_iV = torch::nn::functional::linear(iV, tensors["value_adaptor.weight"], map_get(tensors, "value_adaptor.bias")).view({bsize, seql, nheads, adim}).transpose(1, 2);
	}
	#endif

	#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
	bool not_non_states = states.size() > 0;
	if (not_non_states) {
		auto _h_real_iK = states[0];
		auto _h_real_iV = states[1];
		if (pyt_is_not_none(_h_real_iK)) {
			#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
			seql = nquery + _h_real_iK.size(-1);
			#else
			seql += _h_real_iK.size(-1);
			#endif
			real_iK = at::cat({_h_real_iK, real_iK}, -1);
			real_iV = at::cat({_h_real_iV, real_iV}, 2);
		}
		#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
		else {
			seql = nquery;
		}
		#endif
	}
	#endif

	#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
	auto scores = real_iQ.matmul(real_iK) / sqrt(adim);
	#else
	auto scores = real_iQ.matmul(real_iK);
	#endif

	#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
	auto rel_pemb_weight = map_get(tensors, "rel_pemb.weight");
	bool not_none_rel = ct_is_not_none(rel_pemb_weight);
	if (not_none_rel) {
		auto emb_option = torch::nn::functional::EmbeddingFuncOptions();
		auto padding_idx = map_get(iargs, "rel_pemb.padding_idx", -1);
		if (padding_idx >= 0) {
			emb_option = emb_option.padding_idx(padding_idx);
		}
		rel_pos_cache = map_get(tensors, "rel_pos_cache");
		#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
		if (not_non_states) {
			if (ct_is_none(rel_pos_cache)) {
				rel_pos_cache = get_rel_pos(seql, iargs, tensors["rel_pos"]).narrow(0, seql - nquery, nquery).contiguous();
			}
			scores += real_iQ.permute({2, 0, 1, 3}).contiguous().view({nquery, bsize * nheads, adim}).bmm(torch::nn::functional::embedding(rel_pos_cache, rel_pemb_weight, emb_option).transpose(1, 2)).view({nquery, bsize, nheads, seql}).permute({1, 2, 0, 3});
		}
		else {
			if (ct_is_none(rel_pos_cache)) {
				rel_pos_cache = get_rel_pos(nquery, iargs, tensors["rel_pos"]).contiguous();
			}
			scores += real_iQ.permute({2, 0, 1, 3}).contiguous().view({nquery, bsize * nheads, adim}).bmm(torch::nn::functional::embedding(rel_pos_cache, rel_pemb_weight, emb_option).transpose(1, 2)).view({nquery, bsize, nheads, nquery}).permute({1, 2, 0, 3});
		}
		#else
		if (ct_is_none(rel_pos_cache)) {
			rel_pos_cache = get_rel_pos(seql, iargs, tensors["rel_pos"]).narrow(0, seql - nquery, nquery).contiguous();
		}
		scores += real_iQ.permute({2, 0, 1, 3}).contiguous().view({nquery, bsize * nheads, adim}).bmm(torch::nn::functional::embedding(rel_pos_cache, rel_pemb_weight, emb_option).transpose(1, 2)).view({nquery, bsize, nheads, seql}).permute({1, 2, 0, 3});
		#endif
	}

	scores = scores / sqrt(adim);
	#endif

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
	#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_SATTN
	if (not_non_states) {
	#elif defined(_NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN)
	if (evaluation) {
	#else
	if ((not_non_states) or evaluation) {
	#endif
		rs["real_iK"] = real_iK;
		rs["real_iV"] = real_iV;
	}
	#ifndef _NEUTRON_MODULES_BASE_ATTN_BUILD_CATTN
	if (not_none_rel) {
		rs["rel_pos_cache"] = rel_pos_cache;
	}
	#endif
	return rs;
}
