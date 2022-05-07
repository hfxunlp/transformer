#encoding: utf-8

from cnfg.hyp import enable_tqdm

def non_tqdm(x, *args, **kwargs):

	return x

try:
	from tqdm import tqdm as std_tqdm
except Exception as e:
	std_tqdm = non_tqdm

tqdm = std_tqdm if enable_tqdm else non_tqdm
