import json
import unittest
import numpy as np
import mpmath as mp
from reproduce import OUT, unpack_rom, pack_rom, reciprocal_bits, accum, quantize, initial_minimax


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coeff = unpack_rom((OUT/'coefficients_packed.bin').read_bytes())
        cls.report = json.loads((OUT/'validation.json').read_text())

    def test_packed_table_matches_csv(self):
        csv = np.genfromtxt(OUT/'coefficients.csv',delimiter=',',skip_header=1)[:,3:6].astype(np.int64)
        np.testing.assert_array_equal(self.coeff,csv)
        self.assertEqual(pack_rom(csv),(OUT/'coefficients_packed.bin').read_bytes())

    def test_reject_unsupported_domain(self):
        for value in [0,0x3f7fffff,0x40000000,0x7f800000,0x7fc00000,0xbf800000]:
            with self.assertRaises(ValueError):
                reciprocal_bits(value,self.coeff,0)

    def test_signed_ties_to_even(self):
        for value,want in [('2.5',2),('3.5',4),('-2.5',-2),('-3.5',-4)]:
            self.assertEqual(quantize(mp.mpf(value),0),want)

    def test_scalar_vector_agreement_and_faithfulness(self):
        # Every interval boundary and both endpoints, plus deterministic random inputs.
        rng=np.random.default_rng(2025)
        for model in self.report['models']:
            if not model['bias_q56']:
                continue
            for i in range(128):
                offsets=[0,1,65534,65535]+list(rng.integers(0,65536,size=8))
                for t in offsets:
                    bits=0x3f800000+i*65536+int(t)
                    result=reciprocal_bits(bits,self.coeff,model['bias_q56'],model['mode'])
                    y=result-0x3f000000+2**23
                    mant=bits-0x3f800000+2**23
                    floor,rem=divmod(2**47,mant)
                    self.assertIn(y,[floor,floor+bool(rem)])
                    raw=int(accum(self.coeff,i,model['mode'])[t])+model['bias_q56']
                    ref,r=divmod(raw,2**32)
                    ref+= r>2**31 or (r==2**31 and ref%2)
                    self.assertEqual(y,ref)
            self.assertEqual(reciprocal_bits(0x3f800000,self.coeff,model['bias_q56'],model['mode']),0x3f800000)

    def test_report_coverage_and_target(self):
        for model in self.report['models']:
            self.assertEqual(model['tested_inputs'],2**23)
            if model['bias_q56']:
                self.assertEqual(model['violations_strict_1ulp'],0)
                self.assertEqual(model['faithful_rounding_violations'],0)
                self.assertLess(model['max_absolute_error_ulp'],1)

    def test_remez_alternation(self):
        a=mp.mpf(1)
        c0,c1,c2=initial_minimax(a)
        h=mp.mpf(1)/128
        d=lambda x:-1/(a+x)**2-c1-2*c2*x
        u=mp.findroot(d,(h/10,h*mp.mpf('.4')))
        v=mp.findroot(d,(h*mp.mpf('.6'),h*mp.mpf('.9')))
        errors=[1/(a+x)-c0-c1*x-c2*x*x for x in [0,u,v,h]]
        self.assertLess(max(abs(errors[j]+errors[j+1]) for j in range(3)),mp.mpf('1e-50'))


if __name__=='__main__':
    unittest.main()
