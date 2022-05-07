#include <torch/extension.h>
#include <vector>

at::Tensor lgate_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, int64_t dim, bool inplace=false) {

	torch::Tensor cell;
	if (inplace) {
		cell = igh;
	}
	else {
		cell = igh.clone();
	}
	auto seqlen = cell.size(dim);
	cell.select(dim, 0).addcmul_(init_cell, fgate.select(dim, 0));
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		cell.select(dim, i).addcmul_(cell.select(dim, i - 1), fgate.select(dim, i));
	}

	return cell;
}

std::vector<torch::Tensor> lgate_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, int64_t dim) {

	auto grad_fgate = grad_cell.clone();
	auto grad_igh = grad_cell.clone();
	auto last_index = grad_cell.size(dim) - 1;
	auto acc_grad_cell = grad_cell.select(dim, last_index);
	auto grad_prev_cell = acc_grad_cell * fgate.select(dim, last_index);
	if (last_index > 0) {
		grad_fgate.select(dim, last_index).mul_(cell.select(dim, last_index - 1));
		int64_t i;
		for (i = last_index - 1; i > 0; i--) {
			acc_grad_cell = grad_fgate.select(dim, i).add_(grad_prev_cell);// grad_fgate is initialized as a copy of grad_cell, performing the accumulation directly on grad_fgate is more efficient.
			grad_igh.select(dim, i).add_(grad_prev_cell);
			grad_prev_cell = acc_grad_cell * fgate.select(dim, i);
			acc_grad_cell.mul_(cell.select(dim, i - 1));
		}
		acc_grad_cell = grad_fgate.select(dim, 0).add_(grad_prev_cell);
		grad_igh.select(dim, 0).add_(grad_prev_cell);
		grad_prev_cell = acc_grad_cell * fgate.select(dim, 0);
		acc_grad_cell.mul_(init_cell);
	}
	else {
		grad_fgate.select(dim, last_index).mul_(init_cell);
	}

	return {grad_fgate, grad_igh, grad_prev_cell};
}

std::vector<torch::Tensor> lgate_backward_no_fgate(torch::Tensor grad_cell, torch::Tensor fgate, int64_t dim) {

	auto grad_igh = grad_cell.clone();
	auto last_index = grad_cell.size(dim) - 1;
	auto grad_prev_cell = grad_cell.select(dim, last_index) * fgate.select(dim, last_index);
	int64_t i;
	for (i = last_index - 1; i >= 0; i--) {
		grad_prev_cell = grad_igh.select(dim, i).add_(grad_prev_cell) * fgate.select(dim, i);
	}

	return {grad_igh, grad_prev_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &lgate_forward, "LGate forward");
	m.def("backward", &lgate_backward, "LGate backward");
	m.def("backward_no_fgate", &lgate_backward_no_fgate, "LGate backward (no fgate)");
}
