#encoding: utf-8

import torch
from random import shuffle
from torch.optim import Adam as Optimizer

from loss.base import LabelSmoothingLoss
from lrsch import GoogleLR as LRScheduler
from parallel.base import DataParallelCriterion
from parallel.optm import MultiGPUGradScaler
from parallel.parallelMT import DataParallelMT
from transformer.NMT import NMT
from utils.base import free_cache, get_logger, mkdir, set_random_seed
from utils.contpara import get_model_parameters
from utils.dynbatch import GradientMonitor
from utils.fmt.base import iter_to_str
from utils.fmt.base4torch import load_emb, parse_cuda
from utils.fmt.parser import parse_double_value_tuple
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model, save_states
from utils.state.holder import Holder
from utils.state.pyrand import PyRandomState
from utils.state.thrand import THRandomState
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm
from utils.train.base import getlr, optm_step, optm_step_zero_grad_set_none
from utils.train.dss import dynamic_sample

import cnfg.dynb as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

update_angle = cnfg.update_angle
enc_layer, dec_layer = parse_double_value_tuple(cnfg.nlayer)

def select_function(modin, select_index):

	_sel_m = (list(modin.enc.nets) + list(modin.dec.nets))[select_index]

	return _sel_m.parameters()

grad_mon = GradientMonitor(enc_layer + dec_layer, select_function, module=None, angle_alpha=cnfg.dyn_tol_alpha, num_tol_amin=cnfg.dyn_tol_amin, num_his_record=cnfg.num_dynb_his, num_his_gm=1)

def train(td, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm=32768, nreport=None, save_every=None, chkpf=None, state_holder=None, statesf=None, num_checkpoint=1, cur_checkid=0, report_eva=True, remain_steps=None, save_loss=False, save_checkp_epoch=False, scaler=None):

	sum_loss = part_loss = 0.0
	sum_wd = part_wd = 0
	_done_tokens, _cur_checkid, _cur_rstep, _use_amp = done_tokens, cur_checkid, remain_steps, scaler is not None
	global minerr, minloss, wkdir, save_auto_clean, namin, grad_mon, update_angle
	model.train()
	cur_b, _ls = 1, {} if save_loss else None
	src_grp, tgt_grp = td["src"], td["tgt"]
	for i_d in tqdm(tl, mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[i_d][()])
		seq_o = torch.from_numpy(tgt_grp[i_d][()])
		lo = seq_o.size(1) - 1
		if mv_device:
			seq_batch = seq_batch.to(mv_device, non_blocking=True)
			seq_o = seq_o.to(mv_device, non_blocking=True)
		seq_batch, seq_o = seq_batch.long(), seq_o.long()

		oi = seq_o.narrow(1, 0, lo)
		ot = seq_o.narrow(1, 1, lo).contiguous()
		with torch_autocast(enabled=_use_amp):
			output = model(seq_batch, oi)
			loss = lossf(output, ot)
			if multi_gpu:
				loss = loss.sum()
		loss_add = loss.data.item()

		if scaler is None:
			loss.backward()
		else:
			scaler.scale(loss).backward()

		wd_add = ot.ne(pad_id).int().sum().item()
		loss = output = oi = ot = seq_batch = seq_o = None
		sum_loss += loss_add
		if save_loss:
			_ls[i_d] = loss_add / wd_add
		sum_wd += wd_add
		_done_tokens += wd_add

		_perform_dyn_optm_step, _cos_sim = grad_mon.update(model.module if multi_gpu else model)
		if _perform_dyn_optm_step or (_done_tokens >= tokens_optm):
			if not _perform_dyn_optm_step:
				grad_mon.reset()
			_do_optm_step = True if _cos_sim is None else (_cos_sim <= update_angle)
			if _do_optm_step:
				optm_step(optm, model=model, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer, zero_grad_none=optm_step_zero_grad_set_none)
				lrsch.step()
			else:
				if multi_gpu:
					model.reset_grad()
				else:
					optm.zero_grad(set_to_none=optm_step_zero_grad_set_none)
			_done_tokens = 0
			if _cur_rstep is not None:
				if save_checkp_epoch and (save_every is not None) and (_cur_rstep % save_every == 0) and (chkpf is not None) and (_cur_rstep > 0):
					if num_checkpoint > 1:
						_fend = "_%d.h5" % (_cur_checkid)
						_chkpf = chkpf[:-3] + _fend
						_cur_checkid = (_cur_checkid + 1) % num_checkpoint
					else:
						_chkpf = chkpf
					save_model(model, _chkpf, multi_gpu, print_func=logger.info)
					if statesf is not None:
						save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
				if _do_optm_step:
					_cur_rstep -= 1
					if _cur_rstep <= 0:
						break

		if nreport is not None:
			part_loss += loss_add
			part_wd += wd_add
			if cur_b % nreport == 0:
				if report_eva:
					_leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu, _use_amp)
					logger.info("Average loss over %d tokens: %.3f, valid loss/error: %.3f %.2f" % (part_wd, part_loss / part_wd, _leva, _eeva,))
					if (_eeva < minerr) or (_leva < minloss):
						save_model(model, wkdir + "eva_%.3f_%.2f.h5" % (_leva, _eeva,), multi_gpu, print_func=logger.info, mtyp="ieva" if save_auto_clean else None)
						if statesf is not None:
							save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
						logger.info("New best model saved")
						namin = 0
						if _eeva < minerr:
							minerr = _eeva
						if _leva < minloss:
							minloss = _leva
					free_cache(mv_device)
					model.train()
				else:
					logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd,))
				part_loss = 0.0
				part_wd = 0

		if save_checkp_epoch and (_cur_rstep is None) and (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ntrain):
			if num_checkpoint > 1:
				_fend = "_%d.h5" % (_cur_checkid)
				_chkpf = chkpf[:-3] + _fend
				_cur_checkid = (_cur_checkid + 1) % num_checkpoint
			else:
				_chkpf = chkpf
			save_model(model, _chkpf, multi_gpu, print_func=logger.info)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
		cur_b += 1
	if part_wd != 0.0:
		logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd,))
	return sum_loss / sum_wd, _done_tokens, _cur_checkid, _cur_rstep, _ls

def eva(ed, nd, model, lossf, mv_device, multi_gpu, use_amp=False):
	r = w = 0
	sum_loss = 0.0
	model.eval()
	src_grp, tgt_grp = ed["src"], ed["tgt"]
	with torch_inference_mode():
		for i in tqdm(range(nd), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_o = torch.from_numpy(tgt_grp[bid][()])
			lo = seq_o.size(1) - 1
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
				seq_o = seq_o.to(mv_device, non_blocking=True)
			seq_batch, seq_o = seq_batch.long(), seq_o.long()
			ot = seq_o.narrow(1, 1, lo).contiguous()
			with torch_autocast(enabled=use_amp):
				output = model(seq_batch, seq_o.narrow(1, 0, lo))
				loss = lossf(output, ot)
				if multi_gpu:
					loss = loss.sum()
					trans = torch.cat([outu.argmax(-1).to(mv_device, non_blocking=True) for outu in output], 0)
				else:
					trans = output.argmax(-1)
			sum_loss += loss.data.item()
			data_mask = ot.ne(pad_id)
			correct = (trans.eq(ot) & data_mask).int()
			w += data_mask.int().sum().item()
			r += correct.sum().item()
			correct = data_mask = trans = loss = output = ot = seq_batch = seq_o = None
	w = float(w)
	return sum_loss / w, (w - r) / w * 100.0

def init_fixing(module):

	if hasattr(module, "fix_init"):
		module.fix_init()

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

rid = cnfg.run_id
earlystop = cnfg.earlystop
maxrun = cnfg.maxrun
tokens_optm = cnfg.tokens_optm
done_tokens = 0
batch_report = cnfg.batch_report
report_eva = cnfg.report_eva
use_ams = cnfg.use_ams
cnt_states = cnfg.train_statesf
save_auto_clean = cnfg.save_auto_clean
overwrite_eva = cnfg.overwrite_eva
save_every = cnfg.save_every
start_chkp_save = cnfg.epoch_start_checkpoint_save
epoch_save = cnfg.epoch_save
remain_steps = cnfg.training_steps

wkdir = "".join((cnfg.exp_dir, cnfg.data_id, "/", cnfg.group_id, "/", rid, "/"))
mkdir(wkdir)

chkpf = None
statesf = None
if save_every is not None:
	chkpf = wkdir + "checkpoint.h5"
if cnfg.save_train_state:
	statesf = wkdir + "train.states.t7"

logger = get_logger(wkdir + "train.log")

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)
multi_gpu_optimizer = multi_gpu and cnfg.multi_gpu_optimizer

set_random_seed(cnfg.seed, use_cuda)

td = h5File(cnfg.train_data, "r")
vd = h5File(cnfg.dev_data, "r")

ntrain = td["ndata"][()].item()
nvalid = vd["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

tl = [str(i) for i in range(ntrain)]

logger.info("Design models with seed: %d" % torch.initial_seed())
mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

fine_tune_m = cnfg.fine_tune_m

mymodel = init_model_params(mymodel)
mymodel.apply(init_fixing)
if fine_tune_m is not None:
	logger.info("Load pre-trained model from: " + fine_tune_m)
	mymodel = load_model_cpu(fine_tune_m, mymodel)
	mymodel.apply(load_fixing)

lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction="sum", forbidden_index=cnfg.forbidden_indexes)

if cnfg.src_emb is not None:
	logger.info("Load source embedding from: " + cnfg.src_emb)
	load_emb(cnfg.src_emb, mymodel.enc.wemb.weight, nwordi, cnfg.scale_down_emb, cnfg.freeze_srcemb)
if cnfg.tgt_emb is not None:
	logger.info("Load target embedding from: " + cnfg.tgt_emb)
	load_emb(cnfg.tgt_emb, mymodel.dec.wemb.weight, nwordt, cnfg.scale_down_emb, cnfg.freeze_tgtemb)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	lossf.to(cuda_device, non_blocking=True)

use_amp = cnfg.use_amp and use_cuda
scaler = (MultiGPUGradScaler() if multi_gpu_optimizer else GradScaler()) if use_amp else None

if multi_gpu:
	mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
	lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

if multi_gpu:
	optimizer = mymodel.build_optimizer(Optimizer, lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams, multi_gpu_optimizer=multi_gpu_optimizer, contiguous_parameters=contiguous_parameters)
else:
	optimizer = Optimizer(get_model_parameters(mymodel, contiguous_parameters=contiguous_parameters), lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams)
optimizer.zero_grad(set_to_none=optm_step_zero_grad_set_none)

lrsch = LRScheduler(optimizer, cnfg.isize, cnfg.warm_step, scale=cnfg.lr_scale)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)

state_holder = None if statesf is None and cnt_states is None else Holder(**{"optm": optimizer, "lrsch": lrsch, "pyrand": PyRandomState(), "thrand": THRandomState(use_cuda=use_cuda)})

num_checkpoint = cnfg.num_checkpoint
cur_checkid = 0

tminerr = inf_default

minloss, minerr = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
logger.info("Init lr: %s, Dev Loss/Error: %.3f %.2f" % (" ".join(iter_to_str(getlr(optimizer))), minloss, minerr,))

if fine_tune_m is None:
	save_model(mymodel, wkdir + "init.h5", multi_gpu, print_func=logger.info)
	logger.info("Initial model saved")
else:
	if cnt_states is not None:
		logger.info("Loading training states")
		_remain_states = state_holder.load_state_dict(torch.load(cnt_states))
		remain_steps, cur_checkid = _remain_states["remain_steps"], _remain_states["checkpoint_id"]
		if "training_list" in _remain_states:
			_ctl = _remain_states["training_list"]
		else:
			shuffle(tl)
			_ctl = tl
		tminerr, done_tokens, cur_checkid, remain_steps, _ = train(td, _ctl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, False, False, scaler)
		_ctl = _remain_states = None
		vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
		logger.info("Epoch: 0, train loss: %.3f, valid loss/error: %.3f %.2f" % (tminerr, vloss, vprec,))
		save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.h5" % (tminerr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None)
		if statesf is not None:
			save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		logger.info("New best model saved")

if cnfg.dss_ws is not None and cnfg.dss_ws > 0.0 and cnfg.dss_ws < 1.0:
	dss_ws = int(cnfg.dss_ws * ntrain)
	_Dws = {}
	_prev_Dws = {}
	_crit_inc = {}
	if cnfg.dss_rm is not None and cnfg.dss_rm > 0.0 and cnfg.dss_rm < 1.0:
		dss_rm = int(cnfg.dss_rm * ntrain * (1.0 - cnfg.dss_ws))
	else:
		dss_rm = 0
else:
	dss_ws = 0
	dss_rm = 0
	_Dws = None

namin = 0

for i in range(1, maxrun + 1):
	shuffle(tl)
	free_cache(use_cuda)
	terr, done_tokens, cur_checkid, remain_steps, _Dws = train(td, tl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, dss_ws > 0, i >= start_chkp_save, scaler)
	vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
	logger.info("Epoch: %d, train loss: %.3f, valid loss/error: %.3f %.2f" % (i, terr, vloss, vprec,))

	if (vprec <= minerr) or (vloss <= minloss):
		save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp="eva" if save_auto_clean else None)
		if statesf is not None:
			save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		logger.info("New best model saved")

		namin = 0

		if vprec < minerr:
			minerr = vprec
		if vloss < minloss:
			minloss = vloss

	else:
		if terr < tminerr:
			tminerr = terr
			save_model(mymodel, wkdir + "train_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		elif epoch_save:
			save_model(mymodel, wkdir + "epoch_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)

		namin += 1
		if namin >= earlystop:
			if done_tokens > 0:
				optm_step(optimizer, model=mymodel, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer)
				lrsch.step()
				done_tokens = 0
			logger.info("early stop")
			break

	if remain_steps is not None and remain_steps <= 0:
		logger.info("Last training step reached")
		break

	if dss_ws > 0:
		if _prev_Dws:
			for _key, _value in _Dws.items():
				if _key in _prev_Dws:
					_ploss = _prev_Dws[_key]
					_crit_inc[_key] = (_ploss - _value) / _ploss
			tl = dynamic_sample(_crit_inc, dss_ws, dss_rm)
		_prev_Dws = _Dws

if done_tokens > 0:
	optm_step(optimizer, model=mymodel, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer)
	lrsch.step()

save_model(mymodel, wkdir + "last.h5", multi_gpu, print_func=logger.info)
if statesf is not None:
	save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
logger.info("model saved")

td.close()
vd.close()
