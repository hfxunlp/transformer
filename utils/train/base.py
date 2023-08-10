#encoding: utf-8

from utils.torch.comp import is_fp16_supported

from cnfg.ihyp import optm_step_zero_grad_set_none

def freeze_module(module):

	for p in module.parameters():
		if p.requires_grad:
			p.requires_grad_(False)

def unfreeze_module(module):

	def unfreeze_fixing(mod):

		if hasattr(mod, "fix_unfreeze"):
			mod.fix_unfreeze()

	for p in module.parameters():
		p.requires_grad_(True)

	module.apply(unfreeze_fixing)

def getlr(optm):

	lr = []
	for i, param_group in enumerate(optm.param_groups):
		lr.append(float(param_group["lr"]))

	return lr

def updated_lr(oldlr, newlr):

	rs = False
	for olr, nlr in zip(oldlr, newlr):
		if olr != nlr:
			rs = True
			break

	return rs

def reset_Adam(optm, amsgrad=False):

	for group in optm.param_groups:
		for p in group["params"]:
			state = optm.state[p]
			if len(state) != 0:
				state["step"] = 0
				state["exp_avg"].zero_()
				state["exp_avg_sq"].zero_()
				if amsgrad:
					state["max_exp_avg_sq"].zero_()

def reinit_Adam(optm, amsgrad=False):

	for group in optm.param_groups:
		for p in group["params"]:
			optm.state[p].clear()

def module_train(netin, module, mode=True):

	for net in netin.modules():
		if isinstance(net, module):
			net.train(mode=mode)

	return netin

def optm_step_std(optm, model=None, scaler=None, closure=None, multi_gpu=False, multi_gpu_optimizer=False, zero_grad_none=optm_step_zero_grad_set_none):

	if multi_gpu:
		model.collect_gradients()
	if scaler is None:
		optm.step(closure=closure)
	else:
		scaler.step(optm, closure=closure)
		scaler.update()
	if not multi_gpu_optimizer:
		optm.zero_grad(set_to_none=zero_grad_none)
	if multi_gpu:
		model.update_replicas()

def optm_step_wofp16(optm, model=None, scaler=None, closure=None, multi_gpu=False, multi_gpu_optimizer=False, zero_grad_none=optm_step_zero_grad_set_none):

	if multi_gpu:
		model.collect_gradients()
	optm.step(closure=closure)
	if not multi_gpu_optimizer:
		optm.zero_grad(set_to_none=zero_grad_none)
	if multi_gpu:
		model.update_replicas()

optm_step = optm_step_std if is_fp16_supported else optm_step_wofp16
