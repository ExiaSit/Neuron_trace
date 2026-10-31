import json
from pathlib import Path
import tempfile
import unittest
from web_console.vaa3d import launch_bundle


class Vaa3DTests(unittest.TestCase):
    def test_skeleton_experiment_inputs_are_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);items=[]
            for name in ['skeleton','soma','skeleton_soma']:
                path=root/f'42_{name}.tif';path.write_bytes(b'fixture')
                items.append(dict(group='app2_input',storage_path=str(path)))
            choices=launch_bundle(root,'42',items)['choices']
            self.assertEqual(len(choices),3)
            self.assertTrue(all(c['role']=='mask' and not c['default'] for c in choices))
            self.assertTrue(any('APP2 实际输入' in c['label'] for c in choices))

    def test_manifest_defaults_and_canonical_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def file(name):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'file')
                return str(p)
            raw = file('source/中文 & #+%.tif')
            original = file('original.tif')
            final = file('pruning_swcs/42.swc')
            repaired = file('mask/42.tif')
            alias = root / 'alias.swc'; alias.symlink_to(final)
            manifest = Path(file('manifests/42.json'))
            manifest.write_text(json.dumps(dict(id='42', raw=raw, mask=original)))
            items = [dict(group='mip_comparison', storage_path=str(alias), scope='comparison'),
                     dict(group='pruning_swcs', storage_path=final),
                     dict(group='mask', storage_path=repaired),
                     dict(group='comparison_sources', storage_path=file('42_v081.eswc'), name='v0.81'),
                     dict(group='mask_original', storage_path=raw)]
            data = launch_bundle(root, '42', items)
            self.assertEqual(data['image'], raw)
            self.assertEqual(len(data['choices']), 4)
            self.assertEqual({c['path'] for c in data['choices'] if c['default']}, {final, repaired})
            self.assertEqual(next(c['label'] for c in data['choices'] if c['path'] == final), '最终剪枝 SWC')
            self.assertTrue(any(c['path'] == original for c in data['choices']))

    def test_prepared_fallback_missing_and_foreign_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prep = root/'prepared'; (prep/'img').mkdir(parents=True)
            raw = prep/'img/42_0000.tif'; raw.write_bytes(b'file')
            (root/'manifests').mkdir()
            (root/'manifests/42.json').write_text(json.dumps(dict(id='420',raw=str(raw))))
            self.assertIsNone(launch_bundle(root,'42',[])['image'])
            self.assertEqual(launch_bundle(root,'42',[],prep)['image'],str(raw))
            self.assertIsNone(launch_bundle(root,'420',[],prep)['image'])
            raw.unlink()
            self.assertTrue(launch_bundle(root,'42',[],prep)['unavailable'])
            (root/'manifests/42.json').write_text('[]')
            self.assertIsNone(launch_bundle(root,'42',[])['image'])
            with self.assertRaises(ValueError):launch_bundle(root,'../42',[])

    def test_stage_fallback_and_no_diagnostic_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=[]
            for n in ['diagnostic.swc','app2.swc','gcut.swc']:
                p=root/n;p.write_bytes(b'file');paths.append(str(p))
            groups=['gcut_audits','trace_app2/down_sampled_swcs_app2','gcut_output/selected_for_pruning']
            items=[dict(group=g,storage_path=p) for g,p in zip(groups,paths)]
            choices=launch_bundle(root,'42',items)['choices']
            self.assertEqual([c['path'] for c in choices if c['default']],[paths[2]])
            self.assertFalse(launch_bundle(root,'42',items[:1])['choices'][0]['default'])


if __name__ == '__main__': unittest.main()
