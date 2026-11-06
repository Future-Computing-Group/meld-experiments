"""Cross-platform float jitter of H4 size, injected into embeddings and NLI softmax, then the diag run.
argv: diag.py SEED EMB_ABS NLI_REL <diag args...>"""
import hashlib, importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("diag", sys.argv[1]); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
from meld.merge import signals
SEED, EA, NR = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def rng(*parts):
    return np.random.default_rng(int.from_bytes(hashlib.sha256("|".join((SEED,) + parts).encode()).digest()[:8], "little"))
orig_embed = signals.EmbeddingSimilarity._embed
def embed(self, text):
    v = orig_embed(self, text)
    return v + rng("e", text).uniform(-EA, EA, v.shape) if EA else v
signals.EmbeddingSimilarity._embed = embed
orig = D._probs
def probs(self, a, b):
    p = np.array(orig(self, a, b)) * (1 + rng("n", a, b).normal(0, NR, 3))
    p = tuple(float(x) for x in np.clip(p, 0, 1)); D.FLOAT[a, b] = p
    return p
D._probs = probs
sys.argv = [sys.argv[0]] + sys.argv[5:]
D.main()
