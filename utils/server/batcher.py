#encoding: utf-8

from asyncio import sleep as asleep
from threading import Lock
from time import sleep

from utils.thread import start_thread_with_keeper

from cnfg.server import batcher_maintain_interval as maintain_interval, batcher_wait_interval as wait_interval, batcher_watcher_interval as watcher_interval, thread_keeper_interval

class BatchWrapper:

	def __init__(self, handler, wait_interval=wait_interval, maintain_interval=maintain_interval, watcher_interval=watcher_interval):

		self.handler, self.wait_interval, self.maintain_interval, self.watcher_interval = handler, wait_interval, maintain_interval, watcher_interval
		self.ipool, self.opool, self.rids, self.idpool, self.cid, self.mids, self.ipool_lck, self.opool_lck, self.rids_lck, self.idpool_lck, self.cid_lck, self.mids_lck = {}, {}, set(), set(), 1, set(), Lock(), Lock(), Lock(), Lock(), Lock(), Lock()
		self.opids = tuple()
		self.running = True
		self.t_process, self.t_mnt = start_thread_with_keeper([self.is_running], None, thread_keeper_interval, target=self.processor), start_thread_with_keeper([self.is_running], None, thread_keeper_interval, target=self.maintainer)

	async def __call__(self, x):

		_watcher_interval = self.watcher_interval
		_id = None
		_rs = x
		try:
			_id = self.get_id()
			with self.ipool_lck:
				self.ipool[_id] = x
			while _id in self.ipool:
				await asleep(_watcher_interval)
			while _id in self.rids:
				await asleep(_watcher_interval)
			if _id in self.opool:
				with self.opool_lck:
					_rs = self.opool.pop(_id)
		finally:
			if _id is not None:
				with self.ipool_lck:
					if _id in self.ipool:
						del self.ipool[_id]
				with self.rids_lck:
					if _id in self.rids:
						self.rids.remove(_id)
				with self.opool_lck:
					if _id in self.opool:
						del self.opool[_id]
				with self.idpool_lck:
					self.idpool.add(_id)
				if self.mids and (_id in self.mids):
					with self.mids_lck:
						if _id in self.mids:
							self.mids.remove(_id)

		return _rs

	def processor(self):

		while self.running:
			if self.ipool:
				with self.ipool_lck:
					_hd = self.ipool
					with self.rids_lck:
						self.rids |= set(_hd.keys())
					self.ipool = {}
				_i = list(set(_ for _v in _hd.values() for _ in _v))
				_map = {_k: _v for _k, _v in zip(_i, self.handler(_i))}
				_rs = {_k: [_map.get(_iu, _iu) for _iu in _i] for _k, _i in _hd.items()}
				with self.opool_lck:
					self.opool |= _rs
				with self.rids_lck:
					self.rids -= set(_hd.keys())
			else:
				sleep(self.wait_interval)

	def maintainer(self):

		while self.running:
			if self.opool:
				if self.opids:
					_gc = set()
					with self.opool_lck:
						for _ in self.opids:
							if _ in self.opool:
								del self.opool[_]
								_gc.add(_)
						self.opids = tuple(self.opool.keys())
					if _gc:
						with self.idpool_lck:
							self.idpool |= _gc
			else:
				if (not self.rids) and (not self.ipool):
					if self.mids:
						with self.mids_lck, self.idpool_lck:
							self.idpool |= self.mids
							self.mids.clear()
					else:
						with self.idpool_lck, self.cid_lck, self.ipool_lck, self.rids_lck, self.opool_lck:
							_mids = set(range(1, self.cid)) - self.idpool - set(self.ipool.keys()) - self.rids - set(self.opool.keys())
						if _mids:
							with self.mids_lck:
								self.mids |= _mids
			sleep(self.maintain_interval)

	def get_id_core(self):

		rs = None
		with self.idpool_lck:
			if self.idpool:
				rs = self.idpool.pop()
		if rs is None:
			with self.cid_lck:
				rs = self.cid
				self.cid += 1

		return rs

	def get_id(self):

		rs = self.get_id_core()
		while (rs in self.ipool) or (rs in self.rids) or (rs in self.opool):
			rs = self.get_id_core()

		return rs

	def status(self, mode):

		self.running = mode

	def is_running(self):

		return self.running
