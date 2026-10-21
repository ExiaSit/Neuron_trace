import unittest
import json
from pathlib import Path
import numpy as np
from algorithms.tracing.candidate_evidence import measure, filter_candidates


class CandidateEvidenceTests(unittest.TestCase):
    def test_real_background_regressions(self):
        fixture = Path(__file__).parent/'fixtures'/'soma_background_regression'
        records = json.loads(fixture.with_suffix('.json').read_text())
        with np.load(fixture.with_suffix('.npz')) as data:
            for record in records:
                with self.subTest(id=record['id'], point=record['source_zyx']):
                    key = record['key']
                    result = measure(data[key+'_raw'], data[key+'_mask'], record['local_zyx'])
                    self.assertEqual(result['accepted'], record['expected_accepted'], result)
                    if record['expected_accepted']:
                        self.assertEqual(result['background_policy'], 'large_core_exterior_rescue')
                    else:
                        self.assertEqual(result['background_policy'], 'local')
                        self.assertFalse(result['exterior_rescue_eligible'])

    def sphere(self, peak=70, background=15):
        z,y,x=np.indices((49,49,49))
        body=(z-24)**2+(y-24)**2+(x-24)**2<=6**2
        raw=np.full(body.shape,background,dtype=np.uint8)
        raw[body]=peak
        return raw,body

    def test_dim_compact_soma_with_mask_is_kept(self):
        raw,mask=self.sphere(45,20)
        result=measure(raw,mask,[24,24,24])
        self.assertTrue(result['accepted'],result)

    def test_strong_soma_without_segmentation_is_kept(self):
        raw,mask=self.sphere(170,20)
        result=measure(raw,np.zeros_like(mask),[24,24,24])
        self.assertTrue(result['accepted'],result)
        self.assertTrue(result['strong_raw_exception'])

    def test_large_soma_does_not_supply_its_own_background(self):
        z,y,x=np.indices((65,65,65))
        body=(z-32)**2+(y-32)**2+(x-32)**2<=14**2
        raw=np.where(body,240,15).astype(np.uint8)
        for mask in (body, np.zeros_like(body)):
            result=measure(raw,mask,[32,32,32])
            self.assertTrue(result['accepted'],result)
            self.assertLess(result['background'],20)
            self.assertEqual(result['background_policy'], 'large_core_exterior_rescue')

    def test_small_peak_cannot_use_quiet_distant_background_to_bypass_local_noise(self):
        z,y,x=np.indices((65,65,65))
        distance=(z-32)**2+(y-32)**2+(x-32)**2
        body=distance<=5**2
        raw=np.full(body.shape,15,np.uint8)
        shell=(distance>=10**2)&(distance<=17**2)
        raw[shell & (x<32)]=220
        raw[shell & (x>=32)]=60
        raw[body]=200
        # The mask also only weakly supports this compact local peak.
        support=body & (x<30)
        result=measure(raw,support,[32,32,32])
        self.assertFalse(result['accepted'],result)
        self.assertIn('weak_local_contrast',result['reasons'])
        self.assertFalse(result['exterior_rescue_eligible'])

    def test_distant_bright_signal_does_not_reject_valid_local_soma(self):
        z,y,x=np.indices((65,65,65))
        distance=(z-32)**2+(y-32)**2+(x-32)**2
        body=distance<=6**2
        raw=np.full(body.shape,15,np.uint8)
        raw[(distance>=21**2)&(distance<=32**2)]=210
        raw[body]=100
        result=measure(raw,body,[32,32,32])
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['background_policy'],'local')

    def test_bright_sheet_with_full_mask_is_not_a_soma(self):
        raw=np.full((49,49,49),15,np.uint8);raw[20:29]=180
        result=measure(raw,raw>100,[24,24,24])
        self.assertFalse(result['accepted'])
        self.assertIn('sheet_or_elongated_core',result['reasons'])

    def test_thin_bright_shaft_is_rejected(self):
        raw=np.full((49,49,49),15,np.uint8);raw[:,23:26,23:26]=200
        result=measure(raw,raw>100,[24,24,24])
        self.assertFalse(result['accepted'])
        self.assertIn('no_thick_3d_core',result['reasons'])

    def test_weak_noise_is_rejected_but_explicit_target_protected(self):
        raw,mask=self.sphere(18,10)
        coords=[[24,24,24],[8,8,8]];scores=[5.,float('inf')]
        kept,_,audit=filter_candidates(raw,mask,coords,scores,[8,8,8])
        self.assertEqual(kept,[[8,8,8]])
        self.assertEqual(audit['counts']['rejected'],1)
        self.assertIn('weak_local_contrast',audit['records'][0]['reasons'])

    def test_two_adjacent_real_somas_both_retained(self):
        z,y,x=np.indices((49,49,49))
        body=((z-24)**2+(y-24)**2+(x-17)**2<=5**2)|((z-24)**2+(y-24)**2+(x-31)**2<=5**2)
        raw=np.where(body,100,15).astype(np.uint8)
        coords=[[24,24,17],[24,24,31]]
        kept,_,audit=filter_candidates(raw,body,coords,[float('inf'),5],coords[0])
        self.assertEqual(kept,coords,audit)

    def test_border_compact_soma_not_automatically_deleted(self):
        raw,mask=self.sphere(160,15)
        raw=raw[:,:,22:];mask=mask[:,:,22:]
        result=measure(raw,mask,[24,24,2])
        self.assertTrue(result['accepted'],result)
        self.assertTrue(result['image_boundary_truncated'])


if __name__=='__main__':unittest.main()
