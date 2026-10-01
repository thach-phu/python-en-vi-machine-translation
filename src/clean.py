import re
import unicodedata

# ---------- Cấu hình ----------
BASE = "C:/Users/NITRO/Downloads/PhoMT/PhoMT/detokenization/dev"
MIN_WORDS = 3
MAX_WORDS = 100
MAX_LEN_RATIO = 3.0   # số từ bên dài không quá gấp 3 lần bên ngắn

TONE_MARKS = "\u0300\u0301\u0303\u0309\u0323"  # huyền, sắc, ngã, hỏi, nặng


# ---------- Chuẩn hoá ----------
def fix_tone_position(text):
    nfd = unicodedata.normalize("NFD", text)

    def move(m):
        v1, tone, v2 = m.group(1), m.group(2), m.group(3)
        return v1 + v2 + tone
  
    nfd = re.sub(rf"([oO])([{TONE_MARKS}])([aeAE])", move, nfd)
    nfd = re.sub(rf"(?<![qQ])([uU])([{TONE_MARKS}])([yY])", move, nfd)

    return unicodedata.normalize("NFC", nfd)


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    text = fix_tone_position(text)
    return text


# ---------- Các điều kiện lọc ----------
def not_empty_pair(en, vi):
    return bool(en.strip()) and bool(vi.strip())


def word_count_ok(sentence, min_words, max_words):
    n = len(sentence.split())
    return min_words <= n <= max_words


def length_ratio_ok(en, vi, max_ratio):
    n_en = len(en.split())
    n_vi = len(vi.split())
    if n_en == 0 or n_vi == 0:
        return False
    ratio = max(n_en, n_vi) / min(n_en, n_vi)
    return ratio <= max_ratio


def has_garbled_chars(text):
    if "�" in text:
        return True
    for ch in text:
        if unicodedata.category(ch) == "Cc" and ch not in ("\n", "\t"):
            return True
    return False


def is_valid_pair(en, vi):
    if not not_empty_pair(en, vi):
        return False
    if has_garbled_chars(en) or has_garbled_chars(vi):
        return False
    if not (word_count_ok(en, MIN_WORDS, MAX_WORDS) and word_count_ok(vi, MIN_WORDS, MAX_WORDS)):
        return False
    if not length_ratio_ok(en, vi, MAX_LEN_RATIO):
        return False
    return True


# ---------- Xử lý cả file ----------
def clean_corpus(en_path, vi_path, out_en, out_vi):
    stats = {"total": 0, "invalid": 0, "duplicate": 0, "kept": 0}
    seen = set()

    with open(en_path, encoding="utf-8") as f_en, \
         open(vi_path, encoding="utf-8") as f_vi, \
         open(out_en, "w", encoding="utf-8") as o_en, \
         open(out_vi, "w", encoding="utf-8") as o_vi:

        for en, vi in zip(f_en, f_vi):
            stats["total"] += 1

            en = normalize_text(en)
            vi = normalize_text(vi)

            if not is_valid_pair(en, vi):
                stats["invalid"] += 1
                continue

            key = hash((en, vi))
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            o_en.write(en + "\n")
            o_vi.write(vi + "\n")
            stats["kept"] += 1

    return stats


if __name__ == "__main__":
 stats = clean_corpus(
    f"{BASE}/dev.en", f"{BASE}/dev.vi",
    "dev_clean.en", "dev_clean.vi",
)
print(stats)