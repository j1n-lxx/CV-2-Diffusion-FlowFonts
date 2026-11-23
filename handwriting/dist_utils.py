# -*- coding: utf-8 -*-
"""여러 GPU 병렬 학습 (DistributedDataParallel) 도우미.

    # GPU 1장 (기존과 같음)
    python train.py ...
    # GPU 여러 장: 한 머신에서 4장
    torchrun --standalone --nproc_per_node=4 train.py ...
    # 특정 GPU만:  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py ...

torchrun 이 WORLD_SIZE / RANK / LOCAL_RANK 환경변수를 넣어 주면 자동으로 병렬 모드가 됩니다.
GPU 가 없으면 gloo 백엔드로 CPU 프로세스끼리 돌아서 병렬 코드 경로를 노트북에서도 확인할 수 있습니다.
"""
import os

import torch
import torch.distributed as dist


class Dist:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world > 1
        if torch.cuda.is_available():
            if self.enabled:
                torch.cuda.set_device(self.local)
            self.device = f"cuda:{self.local}" if self.enabled else "cuda"
        elif torch.backends.mps.is_available() and not self.enabled:
            self.device = "mps"
        else:
            self.device = "cpu"
        if self.enabled:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    @property
    def main(self):
        return self.rank == 0

    def wrap(self, model):
        """DDP 로 감쌉니다 (병렬이 아니면 그대로)."""
        if not self.enabled:
            return model
        from torch.nn.parallel import DistributedDataParallel as DDP
        ids = [self.local] if torch.cuda.is_available() else None
        return DDP(model, device_ids=ids)

    def mean(self, x):
        """프로세스들 사이의 평균 (로그용)."""
        if not self.enabled:
            return x
        t = torch.tensor([float(x)], device=self.device if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t)
        return t.item() / self.world

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def close(self):
        if self.enabled:
            dist.barrier()
            dist.destroy_process_group()


def unwrap(model):
    return model.module if hasattr(model, "module") else model
