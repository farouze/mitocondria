import unittest,json,tempfile,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'revised'))
import frozen_figure_and_coverage as coverage
import validation_v3 as validation
import mitoem_external_validation_v2 as original_grid
from resolution_robustness import regrid

class ScientificRegressionChecks(unittest.TestCase):
    def test_global_offset_ignores_summary_median(self):
        m={'global_train_mean_bias_pct':3.65,'calibration_bounds_pct':{'uncorrected':8.45,'global_train_mean':4.63,'occupancy_only_ols':2.66},'summaries':{'global_train_mean':{'median_abs':1.37}}}
        with tempfile.TemporaryDirectory() as d:
            Path(d,'validation_v3_model.json').write_text(json.dumps(m))
            _,offset=coverage.find_bounds_and_offset([d]);self.assertEqual(offset[0],3.65)
    def test_wrong_offset_is_rejected(self):
        raw=np.array([1.,4.,9.]);c=3.65
        df=pd.DataFrame({'id':[1,2,3],'raw':raw,'global':(raw-c)/(1+c/100),'occ':raw/10})
        with self.assertRaises(ValueError):
            coverage.coverage_check(df,{'uncorrected':'raw','global':'global','occ_only':'occ'},{'uncorrected':8.45,'global':4.63,'occ_only':2.66},1.37,'test')
    def test_corrected_residual_algebra(self):
        ref=np.array([2.,7.,9.]);e=np.array([3.,-2.,11.]);hat=np.array([1.,-1.,6.])
        occ=ref*(1+e/100);corrected=occ/(1+hat/100)
        np.testing.assert_allclose(100*(corrected/ref-1),(e-hat)/(1+hat/100))
    def test_conformal_order_and_small_sample(self):
        self.assertEqual(validation.conformal_q(np.arange(1,541)),514.)
        self.assertTrue(np.isinf(validation.conformal_q([1.,2.])))
    def test_interval_inversion(self):
        q=.02661349;pred=10.;ys=np.linspace(9,11,1001)
        np.testing.assert_array_equal(abs(pred/ys-1)<=q,(ys>=pred/(1+q))&(ys<=pred/(1-q)))
    def test_original_phase_unchanged(self):
        rng=np.random.default_rng(24)
        mask=np.pad(rng.random((4,13,17))>.5,1)
        for factor in (2,3):
            np.testing.assert_array_equal(regrid(mask,factor),original_grid.downsample_xy(mask,factor))
    def test_no_ties_in_odd_blocks(self):
        mask=np.pad(np.random.default_rng(3).random((3,8,9))>.5,1)
        np.testing.assert_array_equal(regrid(mask,3,tie_foreground=True),regrid(mask,3,tie_foreground=False))

if __name__=='__main__':unittest.main()
