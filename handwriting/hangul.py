# -*- coding: utf-8 -*-
"""글자 → 조건 변환. 데이터셋(font_ko_v2)의 chars.csv 와 같은 규칙입니다.

한 글자의 조건은 정수 4개입니다.
    [cho, jung, jong, extra]
      한글   : cho 0~18, jung 0~20, jong 0~27 (0 = 받침 없음), extra = NULL_EXTRA
      추가 문자: cho/jung/jong = 각각의 NULL, extra 0~98
'없음'은 각 임베딩의 마지막 인덱스입니다 (Task 2 의 NULL_CLASS 와 같은 방식).

자모로 조건을 주기 때문에 학습 데이터(KS X 1001 2,350자)에 없는 한글 11,172자도 전부
조건으로 만들 수 있습니다. 실제로 잘 그리는지는 eval.py 의 unseen 축으로 확인하세요.
"""
import torch

N_CHO, N_JUNG, N_JONG = 19, 21, 28
N_HANGUL_KS = 2350          # 데이터셋 char_id 0~2349 = KS X 1001 한글, 2350~ = 추가 문자

DIGITS = [chr(c) for c in range(0x30, 0x3A)]
UPPER = [chr(c) for c in range(0x41, 0x5B)]
LOWER = [chr(c) for c in range(0x61, 0x7B)]
ASCII_PUNCT = [chr(c) for c in range(0x21, 0x7F) if not chr(c).isalnum()]
CURLY = ["‘", "’", "“", "”"]
SYMBOL = ["♡"]  # ♡
EXTRA = DIGITS + UPPER + LOWER + ASCII_PUNCT + CURLY + SYMBOL
N_EXTRA = len(EXTRA)  # 99
EXTRA_INDEX = {c: i for i, c in enumerate(EXTRA)}

# 각 칸의 '없음' 토큰 (임베딩 크기 = N + 1)
NULL_CHO, NULL_JUNG, NULL_JONG, NULL_EXTRA = N_CHO, N_JUNG, N_JONG, N_EXTRA
NULL_COND = [NULL_CHO, NULL_JUNG, NULL_JONG, NULL_EXTRA]

# 비슷하게 생긴 문자는 데이터셋에 있는 문자로 바꿔 줍니다.
ALIASES = {"…": "...", "·": ".", "♥": "♡", "❤": "♡", "₩": "\\",
           "！": "!", "？": "?", "，": ",", "．": ".", "。": ".", "～": "~",
           "–": "-", "—": "-", " ": " "}


def is_hangul(ch):
    return 0xAC00 <= ord(ch) <= 0xD7A3


def char_cond(ch):
    """글자 하나 → [cho, jung, jong, extra]. 지원하지 않는 글자면 None."""
    if is_hangul(ch):
        s = ord(ch) - 0xAC00
        return [s // 588, (s % 588) // 28, s % 28, NULL_EXTRA]
    if ch in EXTRA_INDEX:
        return [NULL_CHO, NULL_JUNG, NULL_JONG, EXTRA_INDEX[ch]]
    return None


def normalize(text):
    out = []
    for ch in text:
        out.append(ALIASES.get(ch, ch))
    return "".join(out)


def parse(text):
    """문장 → [(글자, cond 또는 None)]. 띄어쓰기는 (' ', None), 줄바꿈은 ('\n', None), 모르는 글자는 (글자, None)."""
    items = []
    for ch in normalize(text.replace("\r\n", "\n").replace("\r", "\n")):
        if ch == "\n":
            items.append(("\n", None))
        elif ch.isspace():
            items.append((" ", None))
        else:
            items.append((ch, char_cond(ch)))
    return items


def cond_table(chars_csv_rows):
    """chars.csv 행 목록 → (n_chars, 4) long 텐서. char_id 로 인덱싱해서 씁니다."""
    rows = []
    for r in chars_csv_rows:
        cho, jung, jong, ex = int(r["cho"]), int(r["jung"]), int(r["jong"]), int(r["extra_id"])
        rows.append([cho if cho >= 0 else NULL_CHO, jung if jung >= 0 else NULL_JUNG,
                     jong if jong >= 0 else NULL_JONG, ex if ex >= 0 else NULL_EXTRA])
    return torch.tensor(rows, dtype=torch.long)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for ch, c in parse("안녕하세요, 똠방각하! (Hi 2026) “좋아” ♡ …"):
        print(repr(ch), c)
