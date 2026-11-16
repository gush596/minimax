"""Original three-pass construction with constrained-FP16 quantization."""
from pathlib import Path
import importlib.util
import csv
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
spec = importlib.util.spec_from_file_location('original_model', ROOT.parent/'original/reproduce.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
H = mp.mpf(1)/128


def allowed(d):
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[((bits & 0x7c00) != 0x7c00) & ((bits & ((1 << (11-d))-1)) == 0)]
    return np.unique(bits.view(np.float16).astype(np.float64))


def quantize(value, values, d):
    # Locate neighbors with float64, but compare distances in high precision.
    j = int(np.searchsorted(values, float(value)))
    candidates = values[max(0, j-2):min(len(values), j+2)]
    def key(v):
        bits = int(np.float16(v).view(np.uint16))
        return abs(mp.mpf(float(v))-value), (bits >> (11-d)) & 1
    return mp.mpf(float(min(candidates, key=key)))


def residual_extrema(a, c1, c2):
    # The FP16 second pass can produce negative/zero C2. Handle those too.
    der = lambda x: -1/(a+x)**2-c1-2*c2*x
    cuts = [mp.mpf(0), H]
    if c2 > 0:
        turn = c2**(-mp.mpf(1)/3)-a
        if 0 < turn < H:
            cuts.insert(1, turn)
    candidates = list(cuts)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if der(lo)*der(hi) < 0:
            for _ in range(160):
                mid = (lo+hi)/2
                if der(lo)*der(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            candidates.append((lo+hi)/2)
    return [1/(a+x)-c1*x-c2*x*x for x in candidates]


def main():
    OUT.mkdir(exist_ok=True)
    initial = [original.initial_minimax(1+mp.mpf(i)/128) for i in range(128)]
    for d in range(1, 12):
        values = allowed(d)
        with (OUT/f'coefficients_d{d}.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['segment','C0','C1','C2','continuous_max_abs_error','C0_fp16_hex','C1_fp16_hex','C2_fp16_hex'])
            for i, (_, a1, a2) in enumerate(initial):
                a = 1+mp.mpf(i)/128
                c1 = quantize(a1, values, d)
                c2 = quantize(a2+(a1-c1)*128, values, d)
                residuals = residual_extrema(a, c1, c2)
                c0 = quantize((min(residuals)+max(residuals))/2, values, d)
                error = max(abs(c0-v) for v in residuals)
                coeff = np.array([float(c0),float(c1),float(c2)])
                bits = coeff.astype(np.float16).view(np.uint16)
                assert np.all((bits & ((1 << (11-d))-1)) == 0)
                w.writerow([i,*coeff,float(error),*(f'{int(b):04x}' for b in bits)])
        print(f'Three-pass generated d={d}', flush=True)


if __name__ == '__main__':
    main()
