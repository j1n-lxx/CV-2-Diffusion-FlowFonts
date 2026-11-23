# -*- coding: utf-8 -*-
"""평가용 분류기 두 개. Task 2 의 metrics.SmallCNN (MNIST 분류기) 자리에 들어갑니다.

  jamo  : 글자 이미지 → [초성, 중성, 종성, 추가문자] 네 머리. 생성한 글자가 요청한 글자인지(match) 잼.
          사전학습 + 파인튜닝 세트의 train 폰트로 학습하고, 처음 보는 test 폰트로 정확도를 잽니다.
  style : 글자 이미지 → 나눔손글씨 폰트. 한 문장이 한 필체로 나오는지(스타일 일관성) 잼.
          파인튜닝 세트 train 폰트의 '본 글자'로 학습하고 '처음 보는 글자'(train_unseen_chars)로 잽니다.

    python classifier.py --task jamo  --data font_ko_v2 --epochs 3
    python classifier.py --task style --data font_ko_v2 --epochs 5

생성 모델 평가(eval.py)는 이 분류기가 기준 정확도를 넘을 때만 숫자를 믿을 수 있습니다
(Task 2 의 '분류기 98%' 규칙과 같은 이유). 기준: jamo 완전일치 95%, style 90%.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D
import hangul as H
from augment import augment

HEADS = (H.N_CHO + 1, H.N_JUNG + 1, H.N_JONG + 1, H.N_EXTRA + 1)
MIN_ACC = {"jamo": 0.95, "style": 0.90}


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


class GlyphCNN(nn.Module):
    """64 → 32 → 16 → 8 → 4. feature(256차원)는 metrics 의 다양성 계산에도 씁니다."""

    def __init__(self, task="jamo", n_styles=0, width=48):
        super().__init__()
        self.task, self.cfg = task, dict(task=task, n_styles=n_styles, width=width)
        w = width
        self.body = nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w),
                                  nn.Flatten(), nn.Linear(4 * w * 16, 256), nn.ReLU())
        if task == "jamo":
            self.heads = nn.ModuleList([nn.Linear(256, n) for n in HEADS])
        else:
            self.heads = nn.ModuleList([nn.Linear(256, n_styles)])

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        f = self.body(x)
        return [h(f) for h in self.heads]


def targets(task, cond, style):
    return [cond[:, i] for i in range(4)] if task == "jamo" else [style]


@torch.no_grad()
def accuracy(clf, ds, device, n=20000, bs=500, seed=0):
    """각 머리 정확도와 '전부 맞음' 비율."""
    clf.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    ok_all, ok_head, tot = 0, None, 0
    for i in range(0, len(idx), bs):
        x, st, cond = ds.batch(idx[i:i + bs], device)
        outs = clf(x)
        hits = torch.stack([o.argmax(-1) == y for o, y in zip(outs, targets(clf.task, cond, st))])
        ok_head = hits.sum(1) if ok_head is None else ok_head + hits.sum(1)
        ok_all += hits.all(0).sum().item(); tot += x.shape[0]
    return ok_all / tot, (ok_head.float() / tot).tolist()


def load_clf(path, device):
    c = torch.load(path, map_location=device, weights_only=False)
    clf = GlyphCNN(**c["cfg"]).to(device)
    clf.load_state_dict(c["model"]); clf.eval()
    return clf, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="jamo", choices=["jamo", "style"])
    ap.add_argument("--data", default="font_ko_v2")
    ap.add_argument("--split", default="train", help="학습 split (빠른 확인은 dev)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug", type=float, default=1.0, help="증강 세기 (augment.py, 0=끔)")
    ap.add_argument("--aug_p", type=float, default=0.7)
    ap.add_argument("--exclude_file", default="exclude_fonts.txt",
                    help="사전학습 세트에서 뺄 폰트 (학습·평가 모두). 끄려면 none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/clf")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.task == "jamo":
        n_pre = D.n_styles(a.data, "pre")
        ex = D.read_exclude(a.exclude_file)       # 픽셀·장식 폰트와 split 누출 사본 제외
        tr = D.concat(D.load(a.data, "pre", a.split, exclude_fonts=ex),
                      D.load(a.data, "ft", a.split, style_offset=n_pre))
        tests = {"pre_test": D.load(a.data, "pre", "test" if a.split == "train" else a.split, in_memory=False,
                                    exclude_fonts=ex),
                 "ft_test": D.load(a.data, "ft", "test" if a.split == "train" else a.split, in_memory=False)}
        clf = GlyphCNN("jamo")
    else:
        tr = D.load(a.data, "ft", a.split)
        tests = {"ft_unseen_chars": D.load(a.data, "ft", "train_unseen_chars" if a.split == "train" else a.split)}
        clf = GlyphCNN("style", n_styles=D.n_styles(a.data, "ft"))
    clf = clf.to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=a.lr, weight_decay=1e-4)
    per = max(1, len(tr) // a.bs); total = max(1, int(a.epochs * per))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total)
    print(f"device={device}  task={a.task}  학습 {len(tr):,}장  {total}스텝")

    t0, step, g = time.time(), 0, torch.Generator().manual_seed(a.seed)
    while step < total:
        perm = torch.randperm(len(tr), generator=g)
        clf.train()
        for b in range(per):
            if step >= total:
                break
            x, st, cond = tr.batch(perm[b * a.bs:(b + 1) * a.bs], device)
            if a.aug > 0:   # 생성 모델과 같은 증강 → 생성 글자의 흔들림·굵기 차이에 덜 민감하게
                x = augment(x, a.aug, a.aug_p)
            loss = sum(F.cross_entropy(o, y) for o, y in zip(clf(x), targets(a.task, cond, st)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 500 == 0 or step == total:
                print(f"  step {step:>6}/{total}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for name, ds in tests.items():
        acc, heads = accuracy(clf, ds, device)
        res[name] = {"exact": acc, "heads": heads}
        print(f"[{name}] 전부 맞음 {acc:.2%}   머리별 {[f'{h:.3f}' for h in heads]}")
    first = next(iter(res.values()))["exact"]
    path = os.path.join(a.out, f"{a.task}_clf.pt")
    torch.save({"model": clf.state_dict(), "cfg": clf.cfg, "results": res, "args": vars(a)}, path)
    print(f"-> {path}")
    if first < MIN_ACC[a.task]:
        print(f"경고: 정확도 {first:.2%} < 기준 {MIN_ACC[a.task]:.0%}. 이 분류기로 잰 eval 숫자는 믿기 어렵습니다. "
              f"--epochs 를 늘리세요.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
