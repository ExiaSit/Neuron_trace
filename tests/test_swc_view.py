import unittest
import numpy as np
from visualization.swc_view import SwcMipView,crop_xy_axes

class SwcViewTests(unittest.TestCase):
    def test_bbox_margin_and_full_z(self):
        view=SwcMipView((100,150,200),[[40,50,30],[60,70,31]])
        self.assertEqual((view.x0,view.x1,view.y0,view.y1),(25,76,35,86))
        self.assertEqual(view.project(np.zeros((100,150,200)),1).shape,(100,51))
        self.assertEqual(view.extent(1),(24.5,75.5,99.5,-.5))
    def test_crop_precedes_side_projection(self):
        raw=np.zeros((30,100,100));raw[0,0,50]=999;raw[29,50,50]=10
        view=SwcMipView(raw.shape,[[50,50,12]])
        self.assertEqual(view.project(raw,1).max(),10)
        self.assertEqual(view.project(raw,2).max(),10)
        self.assertEqual(view.project(raw,0).max(),10)
    def test_union_includes_removed_or_other_version_nodes(self):
        view=SwcMipView((20,100,100),[[20,20,0],[70,80,19]])
        self.assertEqual((view.x0,view.x1,view.y0,view.y1),(5,86,5,96))
    def test_boundary_empty_and_physical_spacing(self):
        v=SwcMipView((3,40,60),[[0,39,1]])
        self.assertEqual((v.x0,v.x1,v.y0,v.y1),(0,16,24,40))
        v=SwcMipView((3,40,60),[])
        self.assertEqual(v.project(np.zeros((3,40,60)),0).shape,(40,60))
        v=SwcMipView((3,100,100),[[40,40,1]],spacing_xy=(.5,2))
        self.assertEqual((v.x0,v.x1,v.y0,v.y1),(10,71,32,49))
    def test_xy_artist_extent_and_swc_coordinates(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for origin in ['upper','lower']:
            fig,ax=plt.subplots();raw=np.arange(10000).reshape(100,100)
            artist=ax.imshow(raw,origin=origin);line,=ax.plot([40,50],[40,50])
            crop_xy_axes(ax,raw.shape,[[40,40],[50,50]])
            np.testing.assert_array_equal(artist.get_array(),raw[25:66,25:66])
            np.testing.assert_array_equal(line.get_xdata(),[40,50])
            self.assertEqual(ax.get_xlim(),(24.5,65.5))
            self.assertEqual(ax.get_ylim(),(65.5,24.5) if origin=='upper' else (24.5,65.5))
            plt.close(fig)
