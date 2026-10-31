import unittest
import numpy as np
from scipy import ndimage as ndi
from algorithms.mask_repair.connection_thickness import restore
from algorithms.mask_repair.thickness_evidence import guard_mergers, soma_domains


class ThicknessEvidenceTests(unittest.TestCase):
    def shaft(self, amplitude=180, sigma=1.2):
        z,y,x=np.indices((33,33,55));raw=(5+amplitude*np.exp(-((z-16)**2+(y-16)**2)/(2*sigma**2))).astype('uint8')
        mask=np.zeros(raw.shape,bool);mask[16,16,8:47]=True
        path=np.array([[16,16,i] for i in range(8,47)])
        return raw,mask,[path]

    def test_bright_sharp_shaft_grows_without_deleting_input(self):
        raw,mask,paths=self.shaft();before=mask.copy();details={}
        result,added,_=restore(raw,mask,paths,diagnostics=details)
        self.assertGreater(added.sum(),0);self.assertTrue(result[mask].all())
        np.testing.assert_array_equal(mask,before)
        self.assertEqual(details['components_before'],details['components_after'])

    def test_original_dim_shaft_growth_is_restored(self):
        raw,mask,paths=self.shaft(amplitude=20)
        result,added,_=restore(raw,mask,paths)
        self.assertTrue(added.any());self.assertTrue(result[mask].all())

    def test_original_broad_profile_growth_is_restored(self):
        raw,mask,paths=self.shaft(sigma=6)
        _,added,_=restore(raw,mask,paths)
        self.assertTrue(added.any())

    def contact(self, domain_kind):
        mask=np.zeros((9,9,13),bool);mask[4,4,2]=True;mask[4,4,10]=True
        labs,_=ndi.label(mask,np.ones((3,3,3)))
        points1=np.array([[4,4,x] for x in range(3,7)])
        points2=np.array([[4,4,x] for x in range(7,10)])
        added=np.zeros_like(mask);added[tuple(points1.T)]=True;added[tuple(points2.T)]=True
        domains=np.zeros(mask.shape,np.uint16)
        if domain_kind=='same':domains[3:6,3:6,1:12]=1
        if domain_kind=='different':domains[:,:,:7]=1;domains[:,:,7:]=2
        return mask,added,labs,[(points1,1),(points2,2)],domains

    def test_same_soma_can_merge(self):
        mask,added,labels,records,domains=self.contact('same')
        kept,report=guard_mergers(mask,added,labels,records,domains)
        self.assertEqual(report['components_after'],1);self.assertTrue(kept.any())
        self.assertEqual(report['mergers'][0]['status'],'allowed_same_soma')

    def test_outside_soma_is_still_rolled_back(self):
        mask,added,labels,records,domains=self.contact('outside')
        kept,report=guard_mergers(mask,added,labels,records,domains)
        self.assertFalse(kept.any());self.assertEqual(report['components_after'],2)

    def test_two_distinct_soma_domains_do_not_merge(self):
        mask,added,labels,records,domains=self.contact('different')
        kept,report=guard_mergers(mask,added,labels,records,domains)
        self.assertFalse(kept.any());self.assertEqual(report['components_after'],2)

    def test_missing_soma_evidence_keeps_original_guard(self):
        mask,added,labels,records,_=self.contact('same')
        kept,report=guard_mergers(mask,added,labels,records,None)
        self.assertFalse(kept.any());self.assertEqual(report['rollback_voxels'],7)

    def test_one_contact_outside_soma_rejects_the_whole_merge(self):
        mask,added,labels,records,domains=self.contact('same')
        domains[4,4,6]=0
        kept,report=guard_mergers(mask,added,labels,records,domains)
        self.assertFalse(kept.any())
        self.assertTrue(report['mergers'][0]['has_contact_outside_soma'])

    def test_soma_domain_requires_measured_signal_not_bbox(self):
        z,y,x=np.indices((25,25,25));sphere=(z-12)**2+(y-12)**2+(x-12)**2<=25
        raw=np.where(sphere,180,5).astype('uint8');seed=(z-12)**2+(y-12)**2+(x-12)**2<=4
        records=[dict(bbox_zyx=[[2,23]]*3,center_xyz=[12,12,12],core_radius=3,threshold=100)]
        domains=soma_domains(raw,seed,records)
        self.assertTrue(domains.any());self.assertFalse(domains[~sphere].any())
        self.assertEqual(domains[2,2,2],0)

    def test_restore_integrates_same_soma_allowance_and_preserves_source(self):
        z,y,x=np.indices((25,25,25));body=(z-12)**2+(y-12)**2+(x-12)**2<=36
        raw=np.where(body,180,5).astype('uint8');mask=np.zeros_like(body)
        paths=[np.array([[12,12,i] for i in range(7,11)]),np.array([[12,12,i] for i in range(15,19)])]
        for p in paths:mask[tuple(p.T)]=True
        original={};old,_,_=restore(raw,mask,paths,diagnostics=original)
        details={};final,added,_=restore(raw,mask,paths,soma_domain=body.astype('uint16'),diagnostics=details)
        self.assertEqual(original['proposed_voxels'],details['proposed_voxels'])
        self.assertEqual(original['components_after'],2)
        self.assertGreater(original['rollback_voxels'],0)
        self.assertTrue(final[mask].all());self.assertTrue(added.any())
        self.assertEqual(details['components_before'],2);self.assertEqual(details['components_after'],1)
        self.assertEqual(details['mergers'][0]['status'],'allowed_same_soma')

if __name__=='__main__':unittest.main()
