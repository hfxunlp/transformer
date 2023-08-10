#encoding: utf-8

import torch

from utils.func import identity_func

from cnfg.ihyp import allow_fp16_reduction, allow_tf32, enable_torch_check, use_deterministic, use_inference_mode, use_torch_compile

secure_type_map = {torch.float16: torch.float64, torch.float32: torch.float64, torch.uint8: torch.int64, torch.int8: torch.int64, torch.int16: torch.int64, torch.int32: torch.int64}

try:
	if hasattr(torch, "set_float32_matmul_precision"):
		torch.set_float32_matmul_precision("medium" if allow_fp16_reduction else ("high" if allow_tf32 else "highest"))
	torch.backends.cuda.matmul.allow_tf32 = allow_tf32
	torch.backends.cudnn.allow_tf32 = allow_tf32
	torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16_reduction
	torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_fp16_reduction
except Exception as e:
	print(e)

if hasattr(torch.autograd, "set_multithreading_enabled"):
	try:
		torch.autograd.set_multithreading_enabled(True)
	except Exception as e:
		print(e)

# Make cudnn methods deterministic according to: https://pytorch.org/docs/stable/notes/randomness.html#cudnn
_config_cudnn_deterministic_variable = True
if hasattr(torch, "use_deterministic_algorithms"):
	try:
		torch.use_deterministic_algorithms(use_deterministic, warn_only=True)
		_config_cudnn_deterministic_variable = False
	except Exception as e:
		print(e)
if _config_cudnn_deterministic_variable:
	torch.backends.cudnn.deterministic = use_deterministic

torch.backends.cudnn.benchmark = False

if hasattr(torch, "autograd") and hasattr(torch.autograd, "set_detect_anomaly"):
	try:
		torch.autograd.set_detect_anomaly(enable_torch_check)
	except Exception as e:
		print(e)

def all_done_bool(stat, *inputs, **kwargs):

	return stat.all().item()

def all_done_byte(stat, bsize=None, **kwargs):

	return stat.int().sum().item() == (stat.numel() if bsize is None else bsize)

def exist_any_bool(stat):

	return stat.any().item()

def exist_any_byte(stat):

	return stat.int().sum().item() > 0

def torch_all_bool_wodim(x, *inputs, **kwargs):

	return x.all(*inputs, **kwargs)

def torch_all_byte_wodim(x, *inputs, **kwargs):

	return x.int().sum(*inputs, **kwargs).eq(x.numel())

def torch_all_bool_dim(x, dim, *inputs, **kwargs):

	return x.all(dim, *inputs, **kwargs)

def torch_all_byte_dim(x, dim, *inputs, **kwargs):

	return x.int().sum(*inputs, dim=dim, **kwargs).eq(x.size(dim))

def torch_all_bool(x, *inputs, dim=None, **kwargs):

	return x.all(*inputs, **kwargs) if dim is None else x.all(dim, *inputs, **kwargs)

def torch_all_byte(x, *inputs, dim=None, **kwargs):

	return x.int().sum(*inputs, **kwargs).eq(x.numel()) if dim is None else x.int().sum(*inputs, dim=dim, **kwargs).eq(x.size(dim))

def torch_any_bool_wodim(x, *inputs, **kwargs):

	return x.any(*inputs, **kwargs)

def torch_any_byte_wodim(x, *inputs, **kwargs):

	return x.int().sum(*inputs, **kwargs).gt(0)

def torch_any_bool_dim(x, dim, *inputs, **kwargs):

	return x.any(dim, *inputs, **kwargs)

def torch_any_byte_dim(x, dim, *inputs, **kwargs):

	return x.int().sum(*inputs, dim=dim, **kwargs).gt(0)

def torch_any_bool(x, *inputs, dim=None, **kwargs):

	return x.any(*inputs, **kwargs) if dim is None else x.any(dim, *inputs, **kwargs)

def torch_any_byte(x, *inputs, dim=None, **kwargs):

	return x.int().sum(*inputs, **kwargs).gt(0) if dim is None else x.int().sum(*inputs, dim=dim, **kwargs).gt(0)

def flip_mask_bool(mask, dim):

	return mask.to(torch.uint8, non_blocking=True).flip(dim).to(mask.dtype, non_blocking=True)

def flip_mask_byte(mask, dim):

	return mask.flip(dim)

class EmptyAutocast:

	def __init__(self, *inputs, **kwargs):

		self.args, self.kwargs = inputs, kwargs

	def __enter__(self):

		return self

	def __exit__(self, *inputs, **kwargs):

		pass

class EmptyGradScaler:

	def __init__(self, *args, **kwargs):

		self.args, self.kwargs = args, kwargs

	def scale(self, outputs):

		return outputs

	def step(self, optimizer, *args, **kwargs):

		return optimizer.step(*args, **kwargs)

	def update(self, *args, **kwargs):

		pass

def torch_is_autocast_enabled_empty(*args, **kwargs):

	return False

# handling torch.bool
if hasattr(torch, "bool"):
	mask_tensor_type = torch.bool
	secure_type_map[mask_tensor_type] = torch.int64
	nccl_type_map = {torch.bool:torch.uint8}
	all_done = all_done_bool
	exist_any = exist_any_bool
	torch_all = torch_all_bool
	torch_all_dim = torch_all_bool_dim
	torch_all_wodim = torch_all_bool_wodim
	torch_any = torch_any_bool
	torch_any_dim = torch_any_bool_dim
	torch_any_wodim = torch_any_bool_wodim
	flip_mask = flip_mask_bool
else:
	mask_tensor_type = torch.uint8
	nccl_type_map = None
	all_done = all_done_byte
	exist_any = exist_any_byte
	torch_all = torch_all_byte
	torch_all_dim = torch_all_byte_dim
	torch_all_wodim = torch_all_byte_wodim
	torch_any = torch_any_byte
	torch_any_dim = torch_any_byte_dim
	torch_any_wodim = torch_any_byte_wodim
	flip_mask = flip_mask_byte

# handling torch.cuda.amp, fp16 will NOT be really enabled if torch.cuda.amp does not exist (for early versions)
_config_torch_cuda_amp = True
if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
	try:
		from torch.cuda.amp import GradScaler, autocast as torch_autocast
		torch_is_autocast_enabled = torch.is_autocast_enabled
		is_fp16_supported = True
		_config_torch_cuda_amp = False
	except Exception as e:
		print(e)
if _config_torch_cuda_amp:
	torch_autocast, GradScaler, torch_is_autocast_enabled, is_fp16_supported = EmptyAutocast, EmptyGradScaler, torch_is_autocast_enabled_empty, False

# inference mode for torch >= 1.9.0
using_inference_mode = use_inference_mode and hasattr(torch, "inference_mode") and hasattr(torch, "is_inference_mode_enabled")
if using_inference_mode:
	torch_is_inference_mode_enabled, torch_inference_mode = torch.is_inference_mode_enabled, torch.inference_mode
else:
	def torch_is_inference_mode_enabled():

		return not torch.is_grad_enabled()

	def torch_inference_mode(mode=True):

		return torch.set_grad_enabled(not mode)
torch_is_grad_enabled, torch_set_grad_enabled, torch_no_grad = torch.is_grad_enabled, torch.set_grad_enabled, torch.no_grad

torch_compile = torch.compile if hasattr(torch, "compile") and use_torch_compile else identity_func
