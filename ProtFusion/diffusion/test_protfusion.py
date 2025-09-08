import unittest
import torch
import numpy as np
import os
import pandas as pd
import shutil
from argparse import Namespace

# Add ProtFusion to path to allow imports
import sys
sys.path.insert(0, '.')

from ProtFusion.diffusion.utils.dataset import ResidueDataset
from ProtFusion.diffusion.model.resi_score_model import ResiLevelTensorProductScoreModel
from ProtFusion.diffusion.utils.training import loss_func
from ProtFusion.diffusion.diffusion.sampling import ForwardDiffusionKernel
from ProtFusion.diffusion.diffusion.sde import HarmonicSDE as PolymerSDE

class TestProtFusion(unittest.TestCase):
    def setUp(self):
        """Set up a test environment with dummy data."""
        self.test_dir = 'test_data'
        self.pdb_dir = os.path.join(self.test_dir, 'pdb')
        self.emb_dir = os.path.join(self.test_dir, 'emb')
        os.makedirs(os.path.join(self.pdb_dir, 't1'), exist_ok=True)
        os.makedirs(os.path.join(self.emb_dir, 't1'), exist_ok=True)

        # Dummy PDB file
        self.pdb_id = 't1estA.pdb'
        pdb_path = os.path.join(self.pdb_dir, 't1', self.pdb_id)
        with open(pdb_path, 'w') as f:
            f.write("ATOM      1  CA  ILE A   1       8.316  -2.013  11.238  1.00  0.00           C\n")
            f.write("ATOM      2  CA  GLY A   2       7.160  -1.258  11.373  1.00  0.00           C\n")
            f.write("ATOM      3  CA  ALA A   3       7.489   0.198  11.033  1.00  0.00           C\n")
            f.write("ATOM      4  CA  SER A   4       6.623   1.123  11.353  1.00  0.00           C\n")
            f.write("ATOM      5  CA  VAL A   5       6.819   2.518  11.049  1.00  0.00           C\n")

        # Dummy embedding file
        self.seqlen = 5
        emb_path = os.path.join(self.emb_dir, 't1', self.pdb_id)
        node_repr = np.random.rand(self.seqlen, 256)
        edge_repr = np.random.rand(self.seqlen, self.seqlen, 128)
        np.savez(emb_path + '.omegafold_num_recycling.4.npz', node_repr=node_repr, edge_repr=edge_repr)

        # Dummy arguments
        self.args = Namespace(
            pdb_dir=self.pdb_dir, embeddings_dir=self.emb_dir,
            embeddings_key='name', omegafold_num_recycling=4, lm_edge_dim=128, lm_node_dim=256,
            no_edge_embs=False, sde_a=3/(3.8**2), sde_b=0, train_Hf=0.1, inf_step=0.5, train_tmin=0.01,
            inference_mode=False, nmr_dist_embed_dim=32, nmr_loss_weight=0.1, nmr_huber_delta=1.0,
            resi_ns=16, resi_nv=4, resi_ntps=8, resi_ntpv=2, resi_fc_dim=64,
            resi_pos_emb_dim=16, lin_nf=1, lin_self=False, attention=False, sh_lmax=2, order=1,
            t_emb_dim=32, t_emb_type='sinusoidal', radius_emb_type='gaussian', radius_emb_dim=16,
            radius_emb_max=50, tmin=0.001, tmax=1e6, no_radius_sqrt=False, parity=True, resi_conv_layers=2,
            sde_weight=1.0, train_skew=0.0, train_kmin=1, train_cutoff=10.0, train_rmsd_max=0.0,
            cuda_diffuse=False
        )

    def test_pipeline(self):
        """Test the full data loading and model pipeline."""

        original_get_sde = ResidueDataset.get_sde
        def dummy_get_sde(self, i):
            if i == 5:
                sde = PolymerSDE(N=5)
                sde.ts = np.array([0.01, 1.0])
                sde.rmsds = np.array([0.1, 5.0])
                sde.hs = np.array([0.1, 100.0])
                return sde
            return original_get_sde(self, i)

        ResidueDataset.get_sde = dummy_get_sde

        # Dummy split dataframe
        split = pd.DataFrame([{'name': self.pdb_id, 'seqlen': self.seqlen, 'seqres': 'IGASV'}])

        # Instantiate and test dataset
        dataset = ResidueDataset(self.args, split)
        self.assertEqual(len(dataset), 1)
        data = dataset.get(0)
        self.assertFalse(data.skip)
        self.assertTrue('nmr' in data)
        self.assertEqual(data.nmr.edge_index.shape, (2, 1))

        # Apply forward diffusion transform
        transform = ForwardDiffusionKernel(self.args)
        data = transform(data)
        self.assertTrue('score' in data)

        # Instantiate model
        model = ResiLevelTensorProductScoreModel(self.args)

        # Run model
        with torch.no_grad():
            data.pred = model(data)

        # Calculate loss
        sde = data.sde
        loss, base_loss, nmr_loss = loss_func(data, self.args, sde)

        # Assertions
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(nmr_loss.item(), 0)
        print("Test passed: On-the-fly NMR generation and pipeline are working.")

        ResidueDataset.get_sde = original_get_sde

    def tearDown(self):
        """Clean up the test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
