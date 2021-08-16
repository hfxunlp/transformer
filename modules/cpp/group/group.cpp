#include <torch/extension.h>
#include "group_func.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &group_linear_forward, "Group linear forward");
}
