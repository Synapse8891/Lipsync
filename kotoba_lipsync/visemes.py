"""Oculus/VRChat 15 visemes, plus an explicit breath class."""
import numpy as np

LABELS = ("sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou", "breath")
VRM_NAMES = ("A", "I", "U", "E", "O")

# A, I, U, E, O. Consonants are approximate poses, not phonetic ground truth.
VRM_MAP = np.array([
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [.05, .1, 0, 0, 0],
    [.2, .1, 0, 0, 0], [.15, .2, 0, .05, 0], [.25, .1, 0, 0, 0],
    [0, .3, .2, 0, 0], [0, .4, 0, .05, 0], [.08, .1, 0, 0, 0],
    [.15, .1, .1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [.12, 0, .04, 0, 0],
], dtype=np.float32)

JP_PHONES = {
    "a": "aa", "i": "ih", "u": "ou", "e": "E", "o": "oh",
    "A": "aa", "I": "ih", "U": "ou", "E": "E", "O": "oh",
    "p": "PP", "b": "PP", "m": "PP", "my": "PP", "py": "PP", "by": "PP",
    "f": "FF", "fy": "FF", "v": "FF", "t": "DD", "d": "DD", "ty": "DD", "dy": "DD", "ts": "CH",
    "k": "kk", "g": "kk", "ky": "kk", "gy": "kk", "kw": "kk", "gw": "kk", "q": "sil", "cl": "sil",
    "ch": "CH", "sh": "CH", "j": "CH", "s": "SS", "z": "SS",
    "n": "nn", "N": "nn", "ny": "nn", "r": "RR", "ry": "RR", "y": "ih", "w": "ou",
    "h": "breath", "hy": "breath", "pau": "sil", "sil": "sil", "sp": "sil",
}

IPA_PHONES = {
    "p": "PP", "b": "PP", "m": "PP", "ɱ": "PP", "f": "FF", "v": "FF", "ɸ": "FF", "β": "FF",
    "θ": "TH", "ð": "TH", "t": "DD", "d": "DD", "ɾ": "DD", "ʈ": "DD", "ɖ": "DD",
    "k": "kk", "ɡ": "kk", "g": "kk", "q": "kk", "c": "kk", "ɟ": "kk",
    "ʃ": "CH", "ʒ": "CH", "tʃ": "CH", "dʒ": "CH", "tɕ": "CH", "dʑ": "CH", "ç": "CH",
    "s": "SS", "z": "SS", "n": "nn", "ŋ": "nn", "ɲ": "nn", "ɳ": "nn",
    "r": "RR", "ɹ": "RR", "ɻ": "RR", "l": "RR", "ɫ": "RR", "ɚ": "RR", "ɝ": "RR",
    "a": "aa", "ɑ": "aa", "ɐ": "aa", "æ": "aa", "ʌ": "aa", "ə": "aa",
    "ɛ": "E", "e": "E", "eɪ": "E", "ɪ": "ih", "i": "ih", "j": "ih", "ᵻ": "ih",
    "ɔ": "oh", "o": "oh", "ɒ": "oh", "oʊ": "oh", "əʊ": "oh",
    "u": "ou", "ʊ": "ou", "ɯ": "ou", "w": "ou", "y": "ou", "ʉ": "ou",
    "h": "breath", "ɦ": "breath", "ʔ": "sil",
    "aɪ": "aa", "aʊ": "aa", "ɔɪ": "oh", "oɪ": "oh", "iə": "ih", "ɪə": "ih", "ʊə": "ou",
    "ɛə": "E", "eə": "E", "ɛɪ": "E", "ɜ": "aa", "əl": "RR", "ɑɹ": "RR", "ɔɹ": "RR", "ɪɹ": "RR", "ɛɹ": "RR",
    "aɪɚ": "RR", "ɕ": "CH", "ʑ": "CH", "ɴ": "nn", "N": "nn", "ɨ": "ih", "ts": "CH",
}

def phone_to_id(phone: str, language: str) -> int:
    token = phone.strip()
    if language == "ja":
        name = JP_PHONES.get(token)
    else:
        token = token.replace("ː", "").replace("ˑ", "").replace("ˈ", "").replace("ˌ", "").replace("̃", "")
        name = IPA_PHONES.get(token)
    if name is None:
        return -1
    return LABELS.index(name)
