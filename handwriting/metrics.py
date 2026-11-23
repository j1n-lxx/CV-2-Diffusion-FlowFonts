# -*- coding: utf-8 -*-
"""평가 지표. 생성 모델 평가의 표준 지표(FID, IS, Recall, CLIPScore)를 64×64 한글 손글씨에 맞게 옮겼습니다.

주 지표 (보고서 표)
  (FID·precision·recall 의 진짜 묶음 = 생성에 쓴 것과 같은 (필체, 글자) 의 진짜 폰트 글자 1,000개. 글자 구성을 맞춰
   묶음이 작아도 글자 구성 차이가 섞이지 않게 함. 진짜 글자 자신은 FID 0, precision·recall 1)
  fid_clf      FID. 생성 묶음과 진짜 묶음의 분포 거리 (↓). 특징은 Inception 대신 자모 분류기(256차원)
               — DM-Font 의 mFID 처럼 과제 분류기 특징을 씀 (64×64 흑백 글자에는 ImageNet 특징이 맞지 않음).
  is_clf       IS (Inception Score). exp(E_x KL(p(c|x) || p(c))) (↑). 분류기 = 자모 분류기,
               클래스 c = 평가 묶음에 든 글자들. 글자가 또렷하고(확신) 여러 글자가 고르게 나올수록 큼.
               최댓값 = 묶음의 서로 다른 글자 수, 진짜 글자의 같은 값과 나란히 봅니다.
  recall       Recall (Kynkäänniemi 2019, k=3) (↑). 진짜 글자 분포를 생성 묶음이 얼마나 덮는가 = 다양성·모드 붕괴.
  precision    Precision (같은 논문, 같이 나옴) (↑). 생성 글자가 진짜 글자 분포 안에 드는 비율 = 품질.
  clip_score   CLIPScore (Hessel 2021) = 2.5 · max(cos(이미지, 텍스트), 0) (↑). 텍스트 = 요청한 글자
               ('손글씨 글자 "안"'), 다국어 CLIP(xlm-roberta-base-ViT-B-32). 텍스트 → 이미지 일치도.
               CLIP 이 손글씨 한글을 잘 읽는다는 보장이 없으므로 반드시 진짜 글자의 같은 값과 나란히 봅니다.
  clip_rprec   CLIP R-precision. 평가 묶음의 글자 텍스트들 중 CLIP 이 맞는 글자를 1등으로 고르는 비율 (↑).
  (속도)       char_ms (글자 하나 지연), sentence_ms, forwards_per_char (= NFE) 는 eval.py 에서 잼.

보조 지표
  match        요청한 글자(초·중·종성/추가 문자)가 나왔는가 — 자모 분류기 (조건 정확도)
  style_acc    요청한 필체로 나왔는가 — 스타일 분류기
  repeat       같은 글자를 여러 번 쓸 때 매번 조금씩 다른가 (손글씨다움). 고정 글리프 기준선 = 0
  nn_to_train  처음 보는 글자를 그릴 때 학습한 다른 글자를 베끼는가 (Task 2 와 같음)

eval.py --extra 일 때만: fidelity, ssim, lpips, fid(Inception), l2_to_font, seed_variety
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _run(clf, x, device, chunk=500):
    outs, feats = None, []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk].to(device)
        o = [F.softmax(h, -1).cpu() for h in clf(xb)]
        outs = o if outs is None else [torch.cat([a, b]) for a, b in zip(outs, o)]
        feats.append(clf.features(xb).cpu())
    return outs, torch.cat(feats)


def jamo_scores(clf, x, cond, device="cpu", thresh=0.9):
    """(match, 머리별 match, fidelity). match = 네 머리가 전부 맞은 비율."""
    probs, _ = _run(clf, x, device)
    hits = torch.stack([p.argmax(-1) == cond[:, i].cpu() for i, p in enumerate(probs)])
    conf = torch.stack([p.max(-1).values for p in probs]).min(0).values
    return float(hits.all(0).float().mean()), hits.float().mean(1).tolist(), float((conf >= thresh).float().mean())


def style_scores(clf, x, style, groups=None, device="cpu"):
    """(요청 필체 정답률, 문장 일관성). groups: 같은 문장 글자끼리 같은 번호.
    문장 일관성 = 각 문장에서 가장 많이 나온 필체로 분류된 글자의 비율 평균."""
    probs, _ = _run(clf, x, device)
    pred = probs[0].argmax(-1)
    acc = float((pred == style.cpu()).float().mean())
    if groups is None:
        return acc, None
    cons = []
    for gid in groups.unique():
        p = pred[groups.cpu() == gid]
        cons.append(float(torch.bincount(p).max()) / len(p))
    return acc, sum(cons) / len(cons)


def features(clf, x, device="cpu"):
    return _run(clf, x, device)[1]


def seed_variety(clf, xa, xb, device="cpu"):
    """같은 (글자, 필체) 를 시드만 바꿔 두 번 생성했을 때의 거리.
    0 에 가까우면 시드와 무관하게 한 모양만 낸다는 뜻 (모드 붕괴, 1스텝 증류에서 흔함)."""
    fa, fb = features(clf, xa, device), features(clf, xb, device)
    feat = float((fa - fb).norm(dim=1).mean())
    pix = float((xa - xb).reshape(len(xa), -1).norm(dim=1).mean())
    return feat, pix


def style_spread(clf, x, device="cpu"):
    """한 글자를 여러 필체로 그린 묶음 안의 평균 feature 쌍거리. 진짜 폰트 묶음과 비율로 봅니다
    (Task 2 의 '실제 데이터 대비 variety' 와 같은 역할)."""
    f = features(clf, x, device)
    n = len(f)
    return float(torch.cdist(f, f).sum() / (n * (n - 1)))


@torch.no_grad()
def nn_to_train(samples, train_set, device="cpu", chunk=200):
    """Task 2 metrics.nn_to_train 과 같습니다. 반드시 진짜 held-out 데이터의 같은 값과 비교하세요."""
    s = samples.reshape(len(samples), -1).to(device)
    r = train_set.reshape(len(train_set), -1).to(device)
    tot = 0.0
    for i in range(0, len(s), chunk):
        tot += torch.cdist(s[i:i + chunk], r).min(dim=1).values.sum().item()
    return tot / len(s)


def l2_to_target(gen, real):
    """본 글자를 다시 그렸을 때 그 폰트의 진짜 글자와의 픽셀 거리 (작을수록 그 필체에 가깝다)."""
    return float((gen - real).reshape(len(gen), -1).norm(dim=1).mean())


def save_grid(samples, path, nrow=16):
    """[-1,1] (N,1,64,64) → 검은 글자 / 흰 배경 그리드 png (큐레이션 없음)."""
    import numpy as np
    from PIL import Image
    x = ((samples[:nrow * nrow].clamp(-1, 1) + 1) * 127.5).byte().squeeze(1).cpu().numpy()
    n = len(x); rows = (n + nrow - 1) // nrow
    g = np.zeros((rows * 66, nrow * 66), np.uint8)
    for i, im in enumerate(x):
        r, c = divmod(i, nrow)
        g[r * 66 + 1:r * 66 + 65, c * 66 + 1:c * 66 + 65] = im
    Image.fromarray(255 - g).save(path)
    return path


# ================================================================ 선행연구와 맞추는 지표
def _gauss(size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


@torch.no_grad()
def ssim(a, b, chunk=500):
    """[-1,1] (N,1,64,64) 두 묶음의 짝별 SSIM 평균 (Wang 2004, 가우시안 창 11·σ1.5, [0,1] 범위)."""
    w = _gauss()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    tot = 0.0
    for i in range(0, len(a), chunk):
        x = (a[i:i + chunk].float().cpu() + 1) / 2
        y = (b[i:i + chunk].float().cpu() + 1) / 2
        mx, my = F.conv2d(x, w), F.conv2d(y, w)
        sxx = F.conv2d(x * x, w) - mx ** 2
        syy = F.conv2d(y * y, w) - my ** 2
        sxy = F.conv2d(x * y, w) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        tot += m.mean(dim=(1, 2, 3)).sum().item()
    return tot / len(a)


_LPIPS = {}


@torch.no_grad()
def lpips_score(a, b, device="cpu", chunk=250):
    """짝별 LPIPS(AlexNet) 평균. lpips 패키지가 없으면 None (첫 실행 때 가중치를 내려받음)."""
    try:
        import lpips
    except ImportError:
        return None
    if device not in _LPIPS:
        try:
            _LPIPS[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:                       # 가중치를 못 받으면 이 지표만 건너뜀
            print(f"  [LPIPS 건너뜀] AlexNet 가중치를 불러오지 못했습니다: {str(e).splitlines()[-1][:120]}")
            _LPIPS[device] = None
    net, tot = _LPIPS[device], 0.0
    if net is None:
        return None
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk].to(device).float().repeat(1, 3, 1, 1)
        y = b[i:i + chunk].to(device).float().repeat(1, 3, 1, 1)
        tot += net(x, y).sum().item()
    return tot / len(a)


def frechet(fa, fb):
    """두 특징 묶음 (N,D) 의 Fréchet 거리. sqrt(S1 S2) 의 대각합은 고윳값으로 정확히 계산."""
    fa, fb = fa.double(), fb.double()
    mu1, mu2 = fa.mean(0), fb.mean(0)
    s1, s2 = torch.cov(fa.T), torch.cov(fb.T)
    e, v = torch.linalg.eigh(s1)
    r1 = v @ torch.diag(e.clamp(min=0).sqrt()) @ v.T
    tr = torch.linalg.eigvalsh(r1 @ s2 @ r1).clamp(min=0).sqrt().sum()
    return float((mu1 - mu2).pow(2).sum() + torch.trace(s1) + torch.trace(s2) - 2 * tr)


def fid_clf(clf, a, b, device="cpu"):
    """자모 분류기 특징(256차원)으로 잰 FID. 묶음 크기가 같아야 비교할 수 있습니다 (기본 1,000장)."""
    return frechet(features(clf, a, device), features(clf, b, device))


_INC = {}


@torch.no_grad()
def fid_inception(a, b, device="cpu", chunk=50):
    """Inception-v3 pool3 특징 FID (pytorch-fid 와 같은 네트워크). 패키지가 없으면 None."""
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        return None
    if device not in _INC:
        try:
            _INC[device] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        except Exception as e:
            print(f"  [Inception FID 건너뜀] 가중치를 불러오지 못했습니다: {str(e).splitlines()[-1][:120]}")
            _INC[device] = None
    net = _INC[device]
    if net is None:
        return None

    def feats(x):
        out = []
        for i in range(0, len(x), chunk):
            xb = ((x[i:i + chunk].to(device).float() + 1) / 2).repeat(1, 3, 1, 1)     # [0,1], 내부에서 299 로 늘림
            out.append(net(xb)[0].squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out)
    return frechet(feats(a), feats(b))


def repeat_variety(clf, x, device="cpu"):
    """x: (G, K, 1, 64, 64) — G 묶음(같은 필체·같은 글자) × K 번 반복.
    묶음 안 평균 쌍거리 (분류기 특징, 픽셀). 고정 글리프(폰트)는 정의상 0, 클수록 쓸 때마다 다름."""
    G, K = x.shape[:2]
    f = features(clf, x.reshape(G * K, *x.shape[2:]), device).reshape(G, K, -1)
    p = x.reshape(G, K, -1).float().cpu()
    pairs = K * (K - 1)
    feat = float(sum(torch.cdist(f[g], f[g]).sum() for g in range(G)) / (G * pairs))
    pix = float(sum(torch.cdist(p[g], p[g]).sum() for g in range(G)) / (G * pairs))
    return feat, pix


def save_repeat_grid(fixed, gen, path, labels=None):
    """반복 글자 그림: 줄마다 한 글자. 첫 칸 = 고정 글리프(진짜 폰트 글자, 늘 같음), 나머지 = 모델이 K 번 쓴 것.
    fixed (G,1,64,64), gen (G,K,1,64,64), [-1,1]."""
    import numpy as np
    from PIL import Image, ImageDraw
    G, K = gen.shape[:2]
    to = lambda t: (255 - ((t.clamp(-1, 1) + 1) * 127.5).byte().squeeze(0).cpu().numpy())
    gap, cell = 14, 66
    W, H = (K + 1) * cell + gap, G * cell + 22
    g = np.full((H, W), 255, np.uint8)
    for r in range(G):
        y = 22 + r * cell
        g[y + 1:y + 65, 1:65] = to(fixed[r])
        for k in range(K):
            x0 = cell + gap + k * cell
            g[y + 1:y + 65, x0 + 1:x0 + 65] = to(gen[r, k])
    im = Image.fromarray(g).convert("RGB")
    d = ImageDraw.Draw(im)
    d.text((4, 4), "fixed glyph", fill=(120, 120, 120))
    d.text((cell + gap + 4, 4), f"model, same character x{K}", fill=(120, 120, 120))
    d.line([(cell + gap // 2, 20), (cell + gap // 2, H)], fill=(200, 200, 200), width=1)
    im.save(path)
    return path


# ================================================================ 주 지표: IS, Precision/Recall, CLIPScore
@torch.no_grad()
def classifier_is(clf, x, classes, device="cpu"):
    """자모 분류기로 잰 Inception Score.
    classes (K,4): 평가 묶음의 서로 다른 글자 조건. p(c|x) ∝ Π_머리 p_h(c_h|x) 를 K 글자로 정규화.
    반환: IS = exp(E_x KL(p(c|x) || p(c))), 최댓값 K."""
    probs, _ = _run(clf, x, device)
    classes = classes.cpu()
    logp = sum(torch.log(p.clamp_min(1e-12))[:, classes[:, h]] for h, p in enumerate(probs))   # (N,K)
    logp = logp - torch.logsumexp(logp, 1, keepdim=True)
    p = logp.exp()
    log_marg = torch.log(p.mean(0).clamp_min(1e-12))
    kl = (p * (logp - log_marg)).sum(1)
    return float(kl.mean().exp())


@torch.no_grad()
def precision_recall(f_real, f_gen, k=3, chunk=1000):
    """Improved Precision/Recall (Kynkäänniemi et al. 2019). 특징 (N,D) 두 묶음.
    각 점의 k번째 이웃까지 반지름의 공으로 분포(다양체)를 근사.
      precision = 생성 점 중 진짜 다양체 안에 드는 비율,  recall = 진짜 점 중 생성 다양체 안에 드는 비율."""
    f_real, f_gen = f_real.float(), f_gen.float()

    def radii(f):
        r = []
        for i in range(0, len(f), chunk):
            d = torch.cdist(f[i:i + chunk], f)
            r.append(d.kthvalue(k + 1, dim=1).values)            # 자기 자신(0) 다음 k번째
        return torch.cat(r)

    def inside(q, ref, rr):
        hit = []
        for i in range(0, len(q), chunk):
            hit.append((torch.cdist(q[i:i + chunk], ref) <= rr[None]).any(1))
        return float(torch.cat(hit).float().mean())

    return inside(f_gen, f_real, radii(f_real)), inside(f_real, f_gen, radii(f_gen))


CLIP_MODEL = "xlm-roberta-base-ViT-B-32:laion5b_s13b_b90k"
CLIP_PROMPT = '손글씨 글자 "{}"'
_CLIP = {}


def _load_clip(spec, device):
    key = (spec, device)
    if key not in _CLIP:
        try:
            import open_clip
            from torchvision import transforms as TV
            name, _, pre = spec.partition(":")
            model, _, prep = open_clip.create_model_and_transforms(name, pretrained=pre or None, device=device)
            norm = next(t for t in prep.transforms if isinstance(t, TV.Normalize))
            size = model.visual.image_size
            size = size[0] if isinstance(size, (tuple, list)) else size
            _CLIP[key] = (model.eval(), open_clip.get_tokenizer(name), torch.tensor(norm.mean).view(1, 3, 1, 1),
                          torch.tensor(norm.std).view(1, 3, 1, 1), int(size))
        except ImportError:
            print("  [CLIPScore 건너뜀] open_clip_torch·transformers 가 필요합니다 (torch 버전 고정해서 설치, README Setup 참고)")
            _CLIP[key] = None
        except Exception as e:
            print(f"  [CLIPScore 건너뜀] CLIP 을 불러오지 못했습니다: {str(e).splitlines()[-1][:160]}")
            _CLIP[key] = None
    return _CLIP[key]


def _clip_images(x, mean, std, size):
    """[-1,1] 흰 글자/검은 배경 (N,1,64,64) → 검은 글자/흰 바탕, 여백 12.5%, CLIP 입력 크기·정규화."""
    img = 1 - (x.float().clamp(-1, 1) + 1) / 2
    img = F.pad(img, (8, 8, 8, 8), value=1.0)
    img = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (img.repeat(1, 3, 1, 1) - mean) / std


@torch.no_grad()
def clip_score(x, chars, device="cpu", spec=CLIP_MODEL, prompt=CLIP_PROMPT, chunk=250):
    """(CLIPScore, CLIP R-precision). chars: 이미지마다 요청한 글자 문자열.
    CLIPScore = 2.5 · max(cos, 0) 평균 (Hessel et al. 2021). R-precision = 묶음의 서로 다른 글자 텍스트 중 1등이 맞는 비율.
    open_clip 이 없거나 가중치를 못 받으면 (None, None)."""
    m = _load_clip(spec, device)
    if m is None:
        return None, None
    model, tok, mean, std, size = m
    uniq = sorted(set(chars))
    pos = {c: i for i, c in enumerate(uniq)}
    te = []
    for i in range(0, len(uniq), chunk):
        te.append(F.normalize(model.encode_text(tok([prompt.format(c) for c in uniq[i:i + chunk]]).to(device)).float(), dim=-1))
    te = torch.cat(te)
    target = torch.tensor([pos[c] for c in chars], device=device)
    score, top1 = 0.0, 0
    for i in range(0, len(x), chunk):
        xi = _clip_images(x[i:i + chunk].to(device), mean.to(device), std.to(device), size)
        ie = F.normalize(model.encode_image(xi).float(), dim=-1)
        sim = ie @ te.T
        t = target[i:i + chunk]
        score += (2.5 * sim.gather(1, t[:, None]).squeeze(1).clamp_min(0)).sum().item()
        top1 += (sim.argmax(1) == t).sum().item()
    return score / len(x), top1 / len(x)
