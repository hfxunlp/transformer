#include <torch/extension.h>
#include "attn_func.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &_NEUTRON_MODULES_BASE_ATTN_FUNC_NAME, "Self attention forward");
}
