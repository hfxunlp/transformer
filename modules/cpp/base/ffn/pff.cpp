#include <torch/extension.h>
#include "pff_func.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &positionwise_ff_forward, "Positionwise FF forward");
}
