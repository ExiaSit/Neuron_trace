import json
import tempfile
import unittest
from pathlib import Path
from tools.run_connection_experiment import cell_metadata


class ExperimentInputsTests(unittest.TestCase):
    def test_never_uses_60003_coordinates_for_another_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)
            (data/'metadata.json').write_text(json.dumps({'soma_z':47}))
            with self.assertRaises(FileNotFoundError):cell_metadata(data,'97117')

    def test_per_cell_metadata_preserves_actual_z_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp);(data/'metadata').mkdir()
            expected={'soma_z':44.649,'z_resolution':995.0}
            (data/'metadata/97117.json').write_text(json.dumps(expected))
            self.assertEqual(cell_metadata(data,'97117')[0],expected)

    def test_60003_legacy_metadata_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp);(data/'metadata.json').write_text(json.dumps({'soma_z':47}))
            self.assertEqual(cell_metadata(data,'60003')[0]['soma_z'],47)

if __name__=='__main__':unittest.main()
