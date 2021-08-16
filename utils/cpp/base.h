#ifndef _NEUTRON_UTILS_CPP_BASE
#define _NEUTRON_UTILS_CPP_BASE

#include <torch/extension.h>
#include <torch/torch.h>
#include <map>
#include <string>

template <typename T> inline T map_get(std::map<std::string, T> mp, std::string key, T dv=NULL) {
	auto iter = mp.find(key);
	if (iter == mp.end()) {
		return dv;
	}
	else {
		return iter->second;
	}
}

inline torch::Tensor map_get(std::map<std::string, torch::Tensor> mp, std::string key, torch::Tensor dv=torch::Tensor()) {
	auto iter = mp.find(key);
	if (iter == mp.end()) {
		return dv;
	}
	else {
		return iter->second;
	}
}

inline int64_t map_get(std::map<std::string, int64_t> mp, std::string key, int64_t dv=-1) {
	auto iter = mp.find(key);
	if (iter == mp.end()) {
		return dv;
	}
	else {
		return iter->second;
	}
}

inline double map_get(std::map<std::string, double> mp, std::string key, double dv=0.0) {
	auto iter = mp.find(key);
	if (iter == mp.end()) {
		return dv;
	}
	else {
		return iter->second;
	}
}

inline bool map_get(std::map<std::string, bool> mp, std::string key, bool dv=false) {
	auto iter = mp.find(key);
	if (iter == mp.end()) {
		return dv;
	}
	else {
		return iter->second;
	}
}

inline bool is_not_none(torch::Tensor input=torch::Tensor()) {
	if (input.defined() and input.size(-1) > 0) {
		return true;
	}
	else {
		return false;
	}
}

inline bool is_none(torch::Tensor input=torch::Tensor()) {
	if (input.defined() and input.size(-1) > 0) {
		return false;
	}
	else {
		return true;
	}
}

inline bool pyt_is_not_none(torch::Tensor input=torch::Tensor()) {
	return input.size(-1) > 0;
}

inline bool pyt_is_none(torch::Tensor input=torch::Tensor()) {
	return input.size(-1) == 0;
}

inline bool ct_is_not_none(torch::Tensor input=torch::Tensor()) {
	return input.defined();
}

inline bool ct_is_none(torch::Tensor input=torch::Tensor()) {
	return not input.defined();
}

#endif
