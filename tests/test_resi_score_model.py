import unittest
import torch
import os
import sys
from argparse import Namespace

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.resi_score_model import ResiLevelTensorProductScoreModel
from utils.logging import get_logger # Import get_logger

# Configure logger for tests
logger = get_logger(__name__, level="INFO")

class TestResiScoreModelNOESY(unittest.TestCase):

    def setUp(self):
        # Default args for the model
        self.args = Namespace(
            t_emb_type='sinusoidal',
            t_emb_dim=32,
            sh_lmax=2,
            lm_node_dim=256, # Example value
            lm_edge_dim=128, # Example value
            resi_ns=32,
            resi_nv=4,
            resi_ntps=16,
            resi_ntpv=4,
            resi_fc_dim=128,
            radius_emb_dim=50,
            radius_emb_max=50.0,
            no_radius_sqrt=False,
            radius_emb_type='gaussian',
            resi_pos_emb_dim=16,
            order=1,
            lin_nf=1, # default value from ResiLevelTensorProductScoreModel if not in args for conv_layers
            lin_self=False, # default
            attention=False, # default
            parity=True, # default
            # NOESY specific args for the model (matching defaults in model if not provided)
            noesy_emb_dim=32,
            noesy_atom_emb_dim=16,
            noesy_feature_dim=64
        )
        self.model = ResiLevelTensorProductScoreModel(self.args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Testing on device: {self.device}")

    def test_process_noesy_data_empty(self):
        logger.info("Testing _process_noesy_data with empty input")
        num_residues = 10
        processed_noesy = self.model._process_noesy_data(None, num_residues, self.device)
        self.assertEqual(processed_noesy.shape, (num_residues, num_residues, self.args.noesy_feature_dim))
        self.assertTrue(torch.all(processed_noesy == 0))

        processed_noesy_empty_list = self.model._process_noesy_data([], num_residues, self.device)
        self.assertEqual(processed_noesy_empty_list.shape, (num_residues, num_residues, self.args.noesy_feature_dim))
        self.assertTrue(torch.all(processed_noesy_empty_list == 0))

    def test_process_noesy_data_simple_contact(self):
        logger.info("Testing _process_noesy_data with a simple contact")
        num_residues = 20
        # res_idx_from, res_idx_to, peak_id, distance, atom_name_from, atom_name_to
        sample_noesy_contacts = [[0, 5, 1, 3.5, "H", "HB"]]

        processed_noesy = self.model._process_noesy_data(sample_noesy_contacts, num_residues, self.device)
        self.assertEqual(processed_noesy.shape, (num_residues, num_residues, self.args.noesy_feature_dim))

        # Check that the specific edge has non-zero features
        self.assertFalse(torch.all(processed_noesy[0, 5, :] == 0), "Features for contact (0,5) should not be all zero.")
        # Check that a non-contact edge is zero
        self.assertTrue(torch.all(processed_noesy[0, 0, :] == 0)) # Intra-residue, or non-contact
        self.assertTrue(torch.all(processed_noesy[1, 2, :] == 0)) # Unrelated edge

        # Check sum of features (should be non-zero if features were added)
        self.assertNotEqual(torch.sum(processed_noesy).item(), 0)

        # Count non-zero entries along the feature dimension for the specified edge
        # Should be equal to noesy_feature_dim if any feature is non-zero, or some features could be zero.
        # A simpler check is that the sum of absolute values is greater than zero.
        self.assertTrue(torch.sum(torch.abs(processed_noesy[0,5,:])) > 0)


    def test_process_noesy_data_unknown_atom_types(self):
        logger.info("Testing _process_noesy_data with unknown atom types")
        num_residues = 10
        # UNK1 and UNK2 are not in casp13_atom_map, should default to 'H' index
        sample_noesy_contacts = [[0, 1, 1, 2.0, "UNK1", "UNK2"]]

        processed_noesy = self.model._process_noesy_data(sample_noesy_contacts, num_residues, self.device)
        self.assertEqual(processed_noesy.shape, (num_residues, num_residues, self.args.noesy_feature_dim))
        self.assertFalse(torch.all(processed_noesy[0, 1, :] == 0))

        # To verify they defaulted to 'H', we could compare with a known 'H'-'H' contact if embeddings were simple
        # For now, just ensuring it doesn't crash and produces non-zero output is sufficient.

    def test_process_noesy_data_multiple_contacts_same_pair(self):
        logger.info("Testing _process_noesy_data with multiple contacts for the same residue pair")
        num_residues = 10
        sample_noesy_contacts = [
            [0, 1, 1, 2.0, "H", "HB"],
            [0, 1, 2, 2.5, "HB", "H"] # Same pair, different atoms/distance
        ]
        processed_noesy = self.model._process_noesy_data(sample_noesy_contacts, num_residues, self.device)
        self.assertEqual(processed_noesy.shape, (num_residues, num_residues, self.args.noesy_feature_dim))
        # The features for (0,1) should be the sum of the features of the two contacts due to sparse.to_dense()
        # This is harder to test precisely without knowing the exact embedding values.
        # We can check it's non-zero.
        self.assertFalse(torch.all(processed_noesy[0, 1, :] == 0))

        # For a more rigorous test, one might mock the embedding layers or use fixed weights.
        # For now, we ensure it runs and aggregates.

    # Test for the main forward pass (simplified)
    def test_forward_pass_with_and_without_noesy(self):
        logger.info("Testing model forward pass with and without NOESY data")
        num_nodes = 10
        # Create dummy HeteroData
        data = {
            'resi': Namespace(
                x=torch.randn(num_nodes, self.args.lm_node_dim, device=self.device), # node_attr after norm
                pos=torch.randn(num_nodes, 3, device=self.device),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=self.device), # Example edges
                edge_attr_=torch.randn(4, 2 * self.args.lm_edge_dim, device=self.device), # edge_attr_ after norm
                node_t=torch.rand(num_nodes, device=self.device) * 10000, # Example node_t
                num_nodes=num_nodes
            ),
            'sidechain': Namespace( # Assuming sidechain data might be needed by model structure
                 pos=torch.randn(num_nodes * 10, 3, device=self.device) # Placeholder for sidechain atoms
            ),
            'score_norm': torch.rand(num_nodes, device=self.device) # Example score_norm
        }
        # Convert dict to Namespace for attribute access if model expects it (seems like it does)
        data_ns = Namespace(**{k: (v if not isinstance(v, dict) else Namespace(**v)) for k, v in data.items()})


        # Forward pass without NOESY
        try:
            output_no_noesy = self.model(data_ns, noesy_data=None)
            self.assertIsNotNone(output_no_noesy)
            self.assertEqual(output_no_noesy.shape[0], num_nodes) # Should be (num_nodes, 3) or similar
        except Exception as e:
            self.fail(f"Model forward pass without NOESY data failed: {e}")

        # Forward pass with NOESY
        sample_noesy_contacts = [[0, 1, 1, 3.0, "H", "HB"], [1,2,2,4.0,"CA","CB"]] # Using CA/CB which are not H, will default.

        # The model's forward method expects noesy_data to be a list of contact lists if data is a batch
        # If data is a single HeteroData-like object (as prepared here), noesy_data should be a single list of contacts
        try:
            output_with_noesy = self.model(data_ns, noesy_data=sample_noesy_contacts)
            self.assertIsNotNone(output_with_noesy)
            self.assertEqual(output_with_noesy.shape[0], num_nodes)
        except Exception as e:
            self.fail(f"Model forward pass with NOESY data failed: {e}")

        # Ideally, check that output_no_noesy and output_with_noesy are different,
        # but this can be sensitive. For now, just ensure it runs.
        # self.assertFalse(torch.allclose(output_no_noesy, output_with_noesy), "Outputs with and without NOESY should differ.")
        # This might fail if the NOESY contribution is too small or zero due to some logic.
        if output_no_noesy is not None and output_with_noesy is not None: # if both runs succeeded
             self.assertFalse(torch.allclose(output_no_noesy, output_with_noesy),
                             "Outputs with and without NOESY should ideally differ if NOESY has an effect.")
        else:
            logger.warning("Could not compare outputs with/without NOESY as one of the forward passes might have failed or returned None.")


    def test_edge_attr_shape_with_noesy(self):
        logger.info("Testing edge_attr shape modification with NOESY data")
        num_nodes = 5
        num_edges = 4 # Example: 0-1, 1-2, 2-3, 3-4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long, device=self.device)

        # Base dimensions from args
        original_edge_scalar_dim = self.args.t_emb_dim + self.args.radius_emb_dim + self.args.resi_pos_emb_dim + 2 * self.args.lm_edge_dim

        # Mock parts of the data object needed by build_conv_graph and resi_edge_embedding
        data_dict = {
            'resi': Namespace(
                x=torch.randn(num_nodes, self.args.lm_node_dim, device=self.device),
                pos=torch.randn(num_nodes, 3, device=self.device),
                edge_index=edge_index,
                edge_attr_=torch.randn(num_edges, 2 * self.args.lm_edge_dim, device=self.device), # This is edge_attr input to build_conv_graph
                node_t=torch.rand(num_nodes, device=self.device) * 10000,
                num_nodes=num_nodes,
                # For the actual model call later, other parts of 'data' would be needed.
                # But for _process_noesy_data and checking edge_attr shape, this is a start.
            ),
             # Add other minimal data attributes model.forward might expect before edge processing
            'sidechain': Namespace(pos=torch.randn(num_nodes * 10, 3, device=self.device)), # Placeholder
            'score_norm': torch.rand(num_nodes, device=self.device)
        }
        data_ns = Namespace(**{k: (v if not isinstance(v, dict) else Namespace(**v)) for k, v in data_dict.items()})


        # 1. Get edge_attr dimension WITHOUT NOESY
        # We need to call build_conv_graph to get the initial edge_attr
        # Then simulate the concatenation that happens in the main forward method
        node_attr_built, edge_index_built, edge_attr_built_no_noesy, edge_sh_built = \
            self.model.build_conv_graph(data_ns, key='resi', knn=False, edge_pos_emb=True)

        self.assertEqual(edge_attr_built_no_noesy.shape[0], num_edges)
        self.assertEqual(edge_attr_built_no_noesy.shape[1], original_edge_scalar_dim)

        # 2. Get edge_attr dimension WITH NOESY
        sample_noesy_contacts = [[0, 1, 1, 3.0, "H", "HB"]] # Affects the first edge

        # Call _process_noesy_data to get the NOESY feature matrix
        processed_noesy_matrix = self.model._process_noesy_data(sample_noesy_contacts, num_nodes, self.device)
        src_nodes, dst_nodes = edge_index_built[0], edge_index_built[1]
        noesy_features_for_edges = processed_noesy_matrix[src_nodes, dst_nodes]

        # Concatenate like in the forward method
        edge_attr_with_noesy = torch.cat([edge_attr_built_no_noesy, noesy_features_for_edges], dim=-1)

        self.assertEqual(edge_attr_with_noesy.shape[0], num_edges)
        self.assertEqual(edge_attr_with_noesy.shape[1], original_edge_scalar_dim + self.args.noesy_feature_dim)
        logger.info(f"Edge attr dim without NOESY: {edge_attr_built_no_noesy.shape[1]}, with NOESY: {edge_attr_with_noesy.shape[1]}")


if __name__ == '__main__':
    unittest.main()
