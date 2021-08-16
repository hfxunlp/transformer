#include <torch/extension.h>
#include "act_func.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("get_func", &get_func, "Get activation function");
	m.def("forward", &act_forward, "Activation function forward");
}
