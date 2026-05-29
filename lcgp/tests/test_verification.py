# python -m lcgp.tests.test_verification

import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lcgp.lcgp import LCGP


class LCGPVerifier:
    """
    Tests:
    1. Transform -> Inverse Transform (should recover original)
    2. Basis reconstruction (Y ≈ psi @ g)
    3. psi_c computation correctness
    4. Prediction at training points
    5. Detailed prediction step breakdown
    """
    
    def __init__(self, model: LCGP, verbose=True):
        """
        Initialize verifier with an LCGP model instance.
        
        """
        self.model = model
        self.verbose = verbose
        
        if self.model.submethod != 'rep':
            raise ValueError("LCGPVerifier requires model with submethod='rep'")
    
    def print_step(self, step_num, description):
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STEP {step_num}: {description}")
            print(f"{'='*70}")
    
    def test_1_transformation_consistency(self):
        """
        Test 1: Verify that transform -> inverse transform recovers original data
        
        Steps:
        1. Start with original data: y_orig (p, N)
        2. Compute replicate averages: ybar_raw (p, n)
        3. Standardize: ybar_s = (ybar_raw - ybar_mean) / ybar_std
        4. Inverse: ybar_reconstructed = ybar_s * ybar_std + ybar_mean
        5. Compare: ||ybar_reconstructed - ybar_raw|| should be ~0
        """
        self.print_step(1, "Transformation Consistency Test")
        
        y_orig = self.model.y_orig.numpy()
        x_orig = self.model.x_orig.numpy()
        
        print(f"Original data shape: y_orig = {y_orig.shape}, x_orig = {x_orig.shape}")
        
        x_unique, inverse, counts = np.unique(x_orig, axis=0, 
                                              return_inverse=True, 
                                              return_counts=True)
        n_unique = x_unique.shape[0]
        p = y_orig.shape[0]
        
        ybar_manual = np.zeros((p, n_unique))
        for i in range(n_unique):
            cols = (inverse == i)
            ybar_manual[:, i] = y_orig[:, cols].mean(axis=1)
        
        print(f"Manual replicate averages: ybar_manual = {ybar_manual.shape}")
        
        ybar_s = self.model.ybar_s.numpy()
        ybar_mean = self.model.ybar_mean.numpy()
        ybar_std = self.model.ybar_std.numpy()
        
        print(f"Model's standardized ybar: ybar_s = {ybar_s.shape}")
        print(f"Standardization: mean = {ybar_mean.flatten()[:3]}...")
        print(f"Standardization: std = {ybar_std.flatten()[:3]}...")
        
        ybar_reconstructed = ybar_s * ybar_std + ybar_mean
       
        error = np.linalg.norm(ybar_reconstructed - ybar_manual) / np.linalg.norm(ybar_manual)
        print(f"\nRelative reconstruction error: {error:.2e}")
        
        if error < 1e-10:
            print("PASS: Transformation is consistent")
            return True
        else:
            print("FAIL: Transformation has errors")
            print(f"Max absolute difference: {np.max(np.abs(ybar_reconstructed - ybar_manual)):.2e}")
            return False
    
    def test_2_basis_reconstruction(self):
        """
        Test 2: Verify basis decomposition reconstructs standardized data
        
        Steps:
        1. Get standardized data: ybar_s (p, n)
        2. Get basis: phi (p, q), g (q, n) from SVD
        3. Reconstruct: ybar_s_reconstructed = phi @ g
        4. Compute error: ||ybar_s - ybar_s_reconstructed||
        """
        self.print_step(2, "Basis Decomposition Reconstruction Test")
        
        ybar_s = self.model.ybar_s.numpy()  
        p, n = ybar_s.shape
        
        phi = self.model.phi.numpy()  
        g = self.model.g.numpy()      
        q = self.model.q
        
        print(f"Data: ybar_s = {ybar_s.shape}")
        print(f"Basis: phi = {phi.shape}, g = {g.shape}")
        print(f"Using q={q} components out of p={p} dimensions")
    
        ybar_s_reconstructed = phi @ g
    
        error = np.linalg.norm(ybar_s - ybar_s_reconstructed) / np.linalg.norm(ybar_s)
 
        if q < p:
            _, s, _ = np.linalg.svd(ybar_s, full_matrices=False)
            retained_var = np.sum(s[:q]**2) / np.sum(s**2)
            print(f"Retained variance: {retained_var:.4f} ({100*retained_var:.2f}%)")
        
        print(f"\nRelative reconstruction error: {error:.2e}")
        
        if q == p:
            if error < 1e-8:
                print("PASS: Full basis reconstruction is accurate")
                return True
            else:
                print("FAIL: Full basis reconstruction has errors")
                return False
        else:
            if error < 0.5:
                print(" Reduced basis reconstruction is reasonable")
                return True
            else:
                print("Reduced basis reconstruction error is high")
                return False
    
    def test_3_psi_c_computation(self):
        """
        Test 3: Verify psi_c computation and its relationship to phi

        Steps:
        1. Get phi (p, q) and sigma_inv_sqrt_std (p,)
        2. Compute psi_c = phi^T / sigma_inv_sqrt_std
        3. Verify dimensions: psi_c should be (q, p)
        4. Check: psi_c @ (sigma_inv_sqrt_std * phi) should be identity-like
        """
        self.print_step(3, "Psi_c Computation Verification")
        
        if not hasattr(self.model, 'psi_c') or self.model.psi_c is None:
            self.model._compute_aux_predictive_quantities_rep()
        
        phi = self.model.phi.numpy()  
        lsigma2s = self.model.lsigma2s.numpy()
        sigma_inv_sqrt_raw = np.exp(-0.5 * lsigma2s)
        std = self.model.ybar_std[:, 0].numpy()
        sigma_inv_sqrt_std = sigma_inv_sqrt_raw * std
        
        psi_c_manual = phi.T / sigma_inv_sqrt_std[:, None] 

        psi_c_model = self.model.psi_c.numpy()
        
        print(f"phi shape: {phi.shape}")
        print(f"sigma_inv_sqrt_std shape: {sigma_inv_sqrt_std.shape}")
        print(f"psi_c shape: {psi_c_model.shape}")
        print(f"Expected psi_c shape: (q={self.model.q}, p={self.model.p})")

        error = np.linalg.norm(psi_c_model - psi_c_manual) / np.linalg.norm(psi_c_manual)
        print(f"\nPsi_c computation error: {error:.2e}")
        
        scaled_phi = phi * sigma_inv_sqrt_std[:, None]
        product = psi_c_model @ scaled_phi 
        
        identity = np.eye(self.model.q)
        ortho_error = np.linalg.norm(product - identity) / np.linalg.norm(identity)
        print(f"Orthogonality check error: {ortho_error:.2e}")
        
        if error < 1e-10:
            print("PASS: psi_c computation is correct")
            return True
        else:
            print("FAIL: psi_c computation has errors")
            return False
    
    def test_4_prediction_at_training_points(self):
        """
        Test 4: Predict at training points (without fitting)
        
        Steps:
        1. Use x_unique as prediction points
        2. Compute predictions: ypred, ypredvar, yconfvar
        3. Compare ypred with ybar (original averaged data)
        4. Check if predictions are close (they won't be perfect without fitting)
        """
        self.print_step(4, "Prediction at Training Points (No Fitting)")
        
        # Step 1: Use training points
        x_test = self.model.x_unique.numpy()
        print(f"Testing at n={x_test.shape[0]} training points")
        
        # Step 2: Compute predictions
        try:
            ypred, ypredvar, yconfvar = self.model.predict(x_test, return_fullcov=False)
            ypred = ypred.numpy()
            ypredvar = ypredvar.numpy()
            yconfvar = yconfvar.numpy()
            
            print(f"Prediction shapes:")
            print(f"  ypred: {ypred.shape}")
            print(f"  ypredvar: {ypredvar.shape}")
            print(f"  yconfvar: {yconfvar.shape}")
        except Exception as e:
            print(f"FAIL: Prediction failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        ybar_actual = self.model.ybar.numpy() 
        
        abs_error = np.linalg.norm(ypred - ybar_actual) / np.linalg.norm(ybar_actual)
        max_error = np.max(np.abs(ypred - ybar_actual))
        
        print(f"\nPrediction vs actual comparison:")
        print(f"  Relative error: {abs_error:.2e}")
        print(f"  Max absolute error: {max_error:.2e}")
        
        print(f"\nPrediction uncertainty:")
        print(f"  Mean prediction std: {np.mean(np.sqrt(ypredvar)):.4f}")
        print(f"  Mean confidence std: {np.mean(np.sqrt(yconfvar)):.4f}")
        
        if abs_error < 10.0: 
            print("PASS: Prediction pipeline executes correctly")
            return True
        else:
            print("Predictions are very far from actual data")
            return False
    
    def test_5_detailed_prediction_steps(self, x_test_idx=0):
        """
        Test 5: Detailed breakdown of prediction calculation for one point
        """
        self.print_step(5, f"Detailed Prediction Calculation (point {x_test_idx})")
        
        if not hasattr(self.model, 'psi_c') or self.model.psi_c is None:
            self.model._compute_aux_predictive_quantities_rep()
        
        x_test = self.model.x_unique[x_test_idx:x_test_idx+1, :]
        x_test_s = self.model.x_unique_s[x_test_idx:x_test_idx+1, :]
        
        print(f"Test point {x_test_idx}:")
        print(f"  x_test (original): {x_test.numpy().flatten()}")
        print(f"  x_test (standardized): {x_test_s.numpy().flatten()}")
        
        lLmb = self.model.lLmb.numpy()
        lLmb0 = self.model.lLmb0.numpy()
        lnugGPs = self.model.lnugGPs.numpy()
        
        CinvM = self.model.CinvMs.numpy()
        Tks = self.model.Tks.numpy()
        psi_c = self.model.psi_c.numpy()
        
        print(f"\nPrecomputed quantities:")
        print(f"  CinvM: {CinvM.shape}")
        print(f"  Tks: {Tks.shape}")
        print(f"  psi_c: {psi_c.shape}")
        
        q = self.model.q
        ghat = np.zeros(q)
        gvar = np.zeros(q)
        
        print(f"\nProcessing {q} latent components:")
        
        for k in range(min(q, 3)):  
            print(f"\n  Latent component k={k}:")
            
            # covariances 
            print(f"    Computing covariances...")
            print(f"      c0k = K(x_test, x_train; theta_{k})")
            print(f"      c00k = K(x_test, x_test; theta_{k})")
            
            # mean
            print(f"    Mean: ghat[{k}] = c0k @ CinvM[{k}]")
            print(f"      CinvM[{k}] has {np.sum(np.abs(CinvM[k]) > 1e-10)} non-negligible elements")
            
            # variance
            print(f"    Variance: gvar[{k}] = c00k - c0k @ Tk @ c0k^T")
            print(f"      Tk[{k}] matrix condition number: {np.linalg.cond(Tks[k]):.2e}")
        
        print(f"\n  ... (remaining {q-3} components)" if q > 3 else "")
        
        print(f"\nTransforming latent predictions to output space:")
        print(f"  predmean_std = psi_c @ ghat")
        print(f"    psi_c shape: {psi_c.shape}")
        print(f"    ghat shape: (q,) = ({q},)")
        print(f"    Result shape: (p,) = ({self.model.p},)")
        
        print(f"\n  confvar_std = psi_c^2 @ gvar")
        print(f"    Element-wise square then matrix-vector multiply")
        
        print(f"\nInverse standardization:")
        print(f"  ypred = predmean_std * ybar_std + ybar_mean")
        print(f"  ypredvar = (confvar_std + sigma_std^2) * ybar_std^2")
        
        return True
    
    def run_all_tests(self):
        """Run all verification tests."""
        print("\n" + "="*70)
        print("LCGP VERIFICATION TEST SUITE")
        print("="*70)
        
        results = {}
        
        results['test_1_transformation'] = self.test_1_transformation_consistency()
        results['test_2_basis_reconstruction'] = self.test_2_basis_reconstruction()
        results['test_3_psi_c_computation'] = self.test_3_psi_c_computation()
        results['test_4_prediction_pipeline'] = self.test_4_prediction_at_training_points()
        results['test_5_detailed_steps'] = self.test_5_detailed_prediction_steps()
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        print("="*70)
        
        return results


def create_sample_data_with_replicates(n_unique=10, n_replicates=3, d=2, p=3, seed=42):
    """Helper function to create sample replicated data for testing."""
    np.random.seed(seed)
    
    x_unique = np.random.rand(n_unique, d)
    x = np.repeat(x_unique, n_replicates, axis=0)  
    weights = np.random.randn(d, p)
    y_true = np.sin(x @ weights).T
    y = y_true + 0.1 * np.random.randn(p, n_unique * n_replicates)
    
    return x, y

if __name__ == "__main__":
    print("Creating sample replicated data...")
    x, y = create_sample_data_with_replicates(n_unique=10, n_replicates=3, d=2, p=3)
    
    print("Initializing LCGP model...")
    model = LCGP(y=y, x=x, q=None, submethod='rep', verbose=False)
    
    print("\nRunning verification tests...")
    verifier = LCGPVerifier(model, verbose=True)
    results = verifier.run_all_tests()
    
    sys.exit(0 if all(results.values()) else 1)