import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils import bootstrap as _bootstrap

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from utils import incremental_screen as inc


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((inc.HERE/'tests/fixtures/incremental-config.json').read_text())
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def write(self, name, coords):
        p=self.base/name
        lines=['1 1 50 50 10 1 -1']
        for i,(x,y) in enumerate(coords,2):lines.append(f'{i} 3 {x} {y} 10 1 1')
        p.write_text('\n'.join(lines)+'\n');return p
    def test_numeric_names(self):
        self.assertEqual(inc.cell_id('image_120001.tif'),'120001')
        self.assertEqual(inc.cell_id('120001.swc'),'120001')
        with self.assertRaises(ValueError):inc.cell_id('random_120001.tif')
    def test_missing_selected(self):
        r=inc.screen(Path('image_120001.tif'),{},self.cfg)
        self.assertTrue(r['selected']);self.assertEqual(r['reason'],'missing_swc')
    def test_one_sided_selected(self):
        p=self.write('120001.swc',[(70+i,50) for i in range(10)])
        r=inc.screen(Path('120001.tif'),{'120001':[p]},self.cfg)
        self.assertTrue(r['selected'])
    def test_balanced_skipped(self):
        p=self.write('120001.swc',[(50+20*np.cos(a),50+20*np.sin(a)) for a in np.linspace(0,2*np.pi,16,endpoint=False)])
        r=inc.screen(Path('120001.tif'),{'120001':[p]},self.cfg)
        self.assertFalse(r['selected']);self.assertEqual(r['reason'],'not_one_sided_xy')
    def test_any_one_sided_selects_id(self):
        a=self.write('a.swc',[(50+20*np.cos(t),50+20*np.sin(t)) for t in np.linspace(0,2*np.pi,16,endpoint=False)])
        b=self.write('b.swc',[(80+i,50) for i in range(10)])
        self.assertTrue(inc.screen(Path('120001.tif'),{'120001':[a,b]},self.cfg)['selected'])
    def test_invalid_is_error_not_normal_skip(self):
        p=self.base/'bad.swc';p.write_text('bad data')
        r=inc.screen(Path('120001.tif'),{'120001':[p]},self.cfg)
        self.assertFalse(r['selected']);self.assertEqual(r['reason'],'swc_analysis_error')


if __name__=='__main__':unittest.main()
