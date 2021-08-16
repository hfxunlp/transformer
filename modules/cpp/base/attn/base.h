#ifndef _NEUTRON_MODULES_CPP_ATTN_BASE
#define _NEUTRON_MODULES_CPP_ATTN_BASE

#include <torch/extension.h>
#include <torch/torch.h>
#include <map>
#include <string>

inline torch::Tensor get_rel_pos(int64_t length, std::map<std::string, int64_t> iargs, torch::Tensor rel_pos) {

	auto xseql = rel_pos.size(0);
	if (length <= xseql) {
		return rel_pos.narrow(0, 0, length).narrow(1, 0, length);
	}
	else {
		auto _rpm = torch::arange(-length + 1, 1, rel_pos.options()).unsqueeze(0);
		#ifdef _NEUTRON_MODULES_BASE_ATTN_BUILD_RES
		return ((_rpm - _rpm.t()).clamp(iargs["net.clamp_min"], iargs["net.clamp_max"]) + iargs["net.rel_shift"]);
		#else
		return ((_rpm - _rpm.t()).clamp(iargs["clamp_min"], iargs["clamp_max"]) + iargs["rel_shift"]);
		#endif
	}
}

#endif
