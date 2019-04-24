#encoding: utf-8

import sys

import torch
from torch import nn

from torch import optim

from parallel.parallel import DataParallelCriterion
from parallel.parallelMT import DataParallelMT

from utils import *

from lrsch import GoogleLR
from loss import LabelSmoothingLoss

from random import shuffle
from math import sqrt

from tqdm import tqdm

from os import makedirs
from os.path import exists as p_check

import h5py

import cnfg

from transformer.NMT import NMT

def train(td, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu, tokens_optm=32768, nreport=None, save_every=None, chkpf=None, chkpof=None, statesf=None, num_checkpoint=1, cur_checkid=0, report_eva=True, remain_steps=None, save_loss=False, save_checkp_epoch=False):

	sum_loss = 0.0
	sum_wd = 0
	part_loss = 0.0
	part_wd = 0
	_done_tokens = done_tokens
	model.train()
	cur_b = 1
	ndata = len(tl)
	_cur_checkid = cur_checkid
	_cur_rstep = remain_steps
	_ls = {} if save_loss else None

	for i_d, t_d in tqdm(tl):
		seq_batch = torch.from_numpy(td[i_d][:]).long()
		seq_o = torch.from_numpy(td[t_d][:]).long()
		lo = seq_o.size(1) - 1
		if mv_device:
			seq_batch = seq_batch.to(mv_device)
			seq_o = seq_o.to(mv_device)

		if _done_tokens >= tokens_optm:
			optm.zero_grad()
			_done_tokens = 0

		oi = seq_o.narrow(1, 0, lo)
		ot = seq_o.narrow(1, 1, lo).contiguous()
		output = model(seq_batch, oi)
		loss = lossf(output, ot)
		if multi_gpu:
			loss = loss.sum()
		loss_add = loss.data.item()
		sum_loss += loss_add
		wd_add = ot.numel() - ot.eq(0).sum().item()
		if save_loss:
			_ls[(i_d, t_d)] = loss_add / wd_add
		sum_wd += wd_add
		_done_tokens += wd_add
		if nreport is not None:
			part_loss += loss_add
			part_wd += wd_add
			if cur_b % nreport == 0:
				if report_eva:
					_leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu)
					logger.info("Average loss over %d tokens: %.3f, valid loss/error: %.3f %.2f" % (part_wd, part_loss / part_wd, _leva, _eeva))
				else:
					logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd))
				part_loss = 0.0
				part_wd = 0

		# scale the sum of losses down according to the number of tokens adviced by: https://mp.weixin.qq.com/s/qAHZ4L5qK3rongCIIq5hQw, I think not reasonable.
		#loss /= wd_add
		loss.backward()

		if _done_tokens >= tokens_optm:
			if multi_gpu:
				model.collect_gradients()
				optm.step()
				model.update_replicas()
			else:
				optm.step()
			if _cur_rstep is not None:
				if save_checkp_epoch and (save_every is not None) and (_cur_rstep % save_every == 0) and (chkpf is not None) and (_cur_rstep > 0):
					if num_checkpoint > 1:
						_fend = "_%d.t7" % (_cur_checkid)
						_chkpf = chkpf[:-3] + _fend
						if chkpof is not None:
							_chkpof = chkpof[:-3] + _fend
						_cur_checkid = (_cur_checkid + 1) % num_checkpoint
					else:
						_chkpf = chkpf
						_chkpof = chkpof
					save_model(model, _chkpf, multi_gpu)
					if chkpof is not None:
						torch.save(optm.state_dict(), _chkpof)
					if statesf is not None:
						save_states(statesf, tl[cur_b - 1:])
				_cur_rstep -= 1
				if _cur_rstep <= 0:
					break
			lrsch.step()

		if save_checkp_epoch and (_cur_rstep is None) and (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ndata):
			if num_checkpoint > 1:
				_fend = "_%d.t7" % (_cur_checkid)
				_chkpf = chkpf[:-3] + _fend
				if chkpof is not None:
					_chkpof = chkpof[:-3] + _fend
				_cur_checkid = (_cur_checkid + 1) % num_checkpoint
			else:
				_chkpf = chkpf
				_chkpof = chkpof
			#save_model(model, _chkpf, isinstance(model, nn.DataParallel))
			save_model(model, _chkpf, multi_gpu)
			if chkpof is not None:
				torch.save(optm.state_dict(), _chkpof)
			if statesf is not None:
				save_states(statesf, tl[cur_b - 1:])
		cur_b += 1
	if part_wd != 0.0:
		logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd))
	return sum_loss / sum_wd, _done_tokens, _cur_checkid, _cur_rstep, _ls

def eva(ed, nd, model, lossf, mv_device, multi_gpu):
	r = 0
	w = 0
	sum_loss = 0.0
	model.eval()
	with torch.no_grad():
		for i in tqdm(range(nd)):
			bid = str(i)
			seq_batch = torch.from_numpy(ed["i"+bid][:]).long()
			seq_o = torch.from_numpy(ed["t"+bid][:]).long()
			lo = seq_o.size(1) - 1
			if mv_device:
				seq_batch = seq_batch.to(mv_device)
				seq_o = seq_o.to(mv_device)
			ot = seq_o.narrow(1, 1, lo).contiguous()
			output = model(seq_batch, seq_o.narrow(1, 0, lo))
			loss = lossf(output, ot)
			if multi_gpu:
				loss = loss.sum()
				trans = torch.cat([torch.argmax(outu, -1).to(mv_device) for outu in output], 0)
			else:
				trans = torch.argmax(output, -1)
			sum_loss += loss.data.item()
			data_mask = 1 - ot.eq(0)
			correct = torch.gt(trans.eq(ot) + data_mask, 1)
			w += data_mask.sum().item()
			r += correct.sum().item()
	w = float(w)
	return sum_loss / w, (w - r) / w * 100.0

def hook_lr_update(optm, flags):
	for group in optm.param_groups:
		for p in group['params']:
			state = optm.state[p]
			if len(state) != 0:
				state['step'] = 0
				state['exp_avg'].zero_()
				state['exp_avg_sq'].zero_()
				if flags:
					state['max_exp_avg_sq'].zero_()

def tostr(lin):
	return [str(lu) for lu in lin]

def save_states(fname, stl):
	with open(fname, "wb") as f:
		f.write(" ".join([i[0][1:] for i in stl]).encode("utf-8"))
		f.write("\n".encode("utf-8"))

def load_states(fname):
	rs = []
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				for tmpu in tmp.decode("utf-8").split():
					if tmpu:
						rs.append(tmpu)
	return [("i" + tmpu, "t" + tmpu) for tmpu in rs]

def init_fixing(module):

	if "fix_init" in dir(module):
		module.fix_init()

rid = cnfg.run_id
if len(sys.argv) > 1:
	rid = sys.argv[1]

earlystop = cnfg.earlystop

maxrun = cnfg.maxrun

tokens_optm = cnfg.tokens_optm

done_tokens = tokens_optm

batch_report = cnfg.batch_report
report_eva = cnfg.report_eva

use_cuda = cnfg.use_cuda
gpuid = cnfg.gpuid

if use_cuda and torch.cuda.is_available():
	use_cuda = True
	if len(gpuid.split(",")) > 1:
		cuda_device = torch.device(gpuid[:gpuid.find(",")].strip())
		cuda_devices = [int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(",")]
		multi_gpu = True
	else:
		cuda_device = torch.device(gpuid)
		multi_gpu = False
		cuda_devices = None
	torch.cuda.set_device(cuda_device.index)
else:
	use_cuda = False
	cuda_device = False
	multi_gpu = False
	cuda_devices = None

set_random_seed(cnfg.seed, use_cuda)

use_ams = cnfg.use_ams

save_optm_state = cnfg.save_optm_state

save_every = cnfg.save_every
start_chkp_save = cnfg.epoch_start_checkpoint_save

epoch_save = cnfg.epoch_save

remain_steps = cnfg.training_steps

wkdir = "".join(("expm/", cnfg.data_id, "/", rid, "/"))
if not p_check(wkdir):
	makedirs(wkdir)

chkpf = None
chkpof = None
statesf = None
if save_every is not None:
	chkpf = wkdir + "checkpoint.t7"
	if save_optm_state:
		chkpof = wkdir + "checkpoint.optm.t7"
	if cnfg.save_train_state:
		statesf = wkdir + "checkpoint.states"

logger = get_logger(wkdir + "train.log")

td = h5py.File(cnfg.train_data, "r")
vd = h5py.File(cnfg.dev_data, "r")

ntrain = int(td["ndata"][:][0])
nvalid = int(vd["ndata"][:][0])
nwordi = int(td["nwordi"][:][0])
nwordt = int(td["nwordt"][:][0])

logger.info("Design models with seed: %d" % torch.initial_seed())
mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

fine_tune_m = cnfg.fine_tune_m

tl = [("i" + str(i), "t" + str(i)) for i in range(ntrain)]

if fine_tune_m is None:
	mymodel = init_model_params(mymodel)
	mymodel.apply(init_fixing)
else:
	logger.info("Load pre-trained model from: " + fine_tune_m)
	mymodel = load_model_cpu(fine_tune_m, mymodel)

#lw = torch.ones(nwordt).float()
#lw[0] = 0.0
#lossf = nn.NLLLoss(lw, ignore_index=0, reduction='sum')
lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=0, reduction='sum', forbidden_index=cnfg.forbidden_indexes)

if cnfg.src_emb is not None:
	logger.info("Load source embedding from: " + cnfg.src_emb)
	_emb = torch.load(cnfg.src_emb, map_location='cpu')
	if nwordi < _emb.size(0):
		_emb = _emb.narrow(0, 0, nwordi).contiguous()
	if cnfg.scale_down_emb:
		_emb.div_(sqrt(cnfg.isize))
	mymodel.enc.wemb.weight.data = _emb.data
	if cnfg.freeze_srcemb:
		mymodel.enc.wemb.weight.requires_grad_(False)
	else:
		mymodel.enc.wemb.weight.requires_grad_(True)
	_emb = None
if cnfg.tgt_emb is not None:
	logger.info("Load target embedding from: " + cnfg.tgt_emb)
	_emb = torch.load(cnfg.tgt_emb, map_location='cpu')
	if nwordt < _emb.size(0):
		_emb = _emb.narrow(0, 0, nwordt).contiguous()
	if cnfg.scale_down_emb:
		_emb.div_(sqrt(cnfg.isize))
	mymodel.dec.wemb.weight.data = _emb
	if cnfg.freeze_tgtemb:
		mymodel.dec.wemb.weight.requires_grad_(False)
	else:
		mymodel.dec.wemb.weight.requires_grad_(True)
	_emb = None

if use_cuda:
	mymodel.to(cuda_device)
	lossf.to(cuda_device)

# lr will be over written by GoogleLR before used
optimizer = optim.Adam(mymodel.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=cnfg.weight_decay, amsgrad=use_ams)

if multi_gpu:
	#mymodel = nn.DataParallel(mymodel, device_ids=cuda_devices, output_device=cuda_device.index)
	mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
	lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

fine_tune_state = cnfg.fine_tune_state
if fine_tune_state is not None:
	logger.info("Load optimizer state from: " + fine_tune_state)
	optimizer.load_state_dict(torch.load(fine_tune_state))

lrsch = GoogleLR(optimizer, cnfg.isize, cnfg.warm_step)
lrsch.step()

num_checkpoint = cnfg.num_checkpoint
cur_checkid = 0

tminerr = float("inf")

minloss, minerr = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
logger.info("".join(("Init lr: ", ",".join(tostr(getlr(optimizer))), ", Dev Loss/Error: %.3f %.2f" % (minloss, minerr))))

if fine_tune_m is None:
	save_model(mymodel, wkdir + "init.t7", multi_gpu)
	logger.info("Initial model saved")
else:
	cnt_states = cnfg.train_statesf
	if (cnt_states is not None) and p_check(cnt_states):
		logger.info("Continue last epoch")
		tminerr, done_tokens, cur_checkid, remain_steps, _ = train(td, load_states(cnt_states), vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, tokens_optm, batch_report, save_every, chkpf, chkpof, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, False, False)
		vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
		logger.info("Epoch: 0, train loss: %.3f, valid loss/error: %.3f %.2f" % (tminerr, vloss, vprec))
		save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.t7" % (tminerr, vloss, vprec), multi_gpu)
		if save_optm_state:
			torch.save(optimizer.state_dict(), wkdir + "train_0_%.3f_%.3f_%.2f.optm.t7" % (tminerr, vloss, vprec))
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
	terr, done_tokens, cur_checkid, remain_steps, _Dws = train(td, tl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, tokens_optm, batch_report, save_every, chkpf, chkpof, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, dss_ws > 0, i >= start_chkp_save)
	vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
	logger.info("Epoch: %d, train loss: %.3f, valid loss/error: %.3f %.2f" % (i, terr, vloss, vprec))

	if (vprec <= minerr) or (vloss <= minloss):
		save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.t7" % (i, terr, vloss, vprec), multi_gpu)
		if save_optm_state:
			torch.save(optimizer.state_dict(), wkdir + "eva_%d_%.3f_%.3f_%.2f.optm.t7" % (i, terr, vloss, vprec))
		logger.info("New best model saved")

		namin = 0

		if vprec < minerr:
			minerr = vprec
		if vloss < minloss:
			minloss = vloss

	else:
		if terr < tminerr:
			tminerr = terr
			save_model(mymodel, wkdir + "train_%d_%.3f_%.3f_%.2f.t7" % (i, terr, vloss, vprec), multi_gpu)
			if save_optm_state:
				torch.save(optimizer.state_dict(), wkdir + "train_%d_%.3f_%.3f_%.2f.optm.t7" % (i, terr, vloss, vprec))
		elif epoch_save:
			save_model(mymodel, wkdir + "epoch_%d_%.3f_%.3f_%.2f.t7" % (i, terr, vloss, vprec), multi_gpu)

		namin += 1
		if namin >= earlystop:
			if done_tokens > 0:
				if multi_gpu:
					mymodel.collect_gradients()
				optimizer.step()
				#lrsch.step()
				done_tokens = 0
				#optimizer.zero_grad()
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

	#oldlr = getlr(optimizer)
	#lrsch.step(terr)
	#newlr = getlr(optimizer)
	#if updated_lr(oldlr, newlr):
		#logger.info("".join(("lr update from: ", ",".join(tostr(oldlr)), ", to: ", ",".join(tostr(newlr)))))
		#hook_lr_update(optimizer, use_ams)

if done_tokens > 0:
	if multi_gpu:
		mymodel.collect_gradients()
	optimizer.step()
	#lrsch.step()
	#done_tokens = 0
	#optimizer.zero_grad()

save_model(mymodel, wkdir + "last.t7", multi_gpu)
if save_optm_state:
	torch.save(optimizer.state_dict(), wkdir + "last.optm.t7")
logger.info("model saved")

td.close()
vd.close()
