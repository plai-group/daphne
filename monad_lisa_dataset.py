import torch
import numpy as np
from torch.distributions import Normal, Beta, Bernoulli, Uniform, Gamma
from graphical_simulator_inference.models.measure import RealSpace, Sample
from graphical_simulator_inference.models.primitives import LATENT, OBSERVED
from graphical_simulator_inference.datasets.graphical_dataset import GraphicalDataset
from graphical_simulator_inference.models.measure import marginal_edges, flatten_samples
from collections import OrderedDict

class MonadLisaDataset(GraphicalDataset):
    """Auto-generated GSDM dataset from Daphne program."""

    def __init__(self, config, seed, is_test):
        super().__init__(config, seed, is_test)
        self.states_and_edges = [self.sample_states_and_edges(marg_prop=config.data.marg_prop) for _ in range(config.data.n_states)]

    @property
    def spaces(self):
        return [RealSpace('sample10', torch.Size([1]), exchangeable='none'), RealSpace('sample34', torch.Size([1]), exchangeable='none'), RealSpace('sample24', torch.Size([1]), exchangeable='none'), RealSpace('observe15', torch.Size([1]), exchangeable='none'), RealSpace('sample12', torch.Size([1]), exchangeable='none'), RealSpace('sample5', torch.Size([1]), exchangeable='none'), RealSpace('sample27', torch.Size([1]), exchangeable='none'), RealSpace('sample28', torch.Size([1]), exchangeable='none'), RealSpace('sample0', torch.Size([1]), exchangeable='none'), RealSpace('sample25', torch.Size([1]), exchangeable='none'), RealSpace('sample17', torch.Size([1]), exchangeable='none'), RealSpace('sample7', torch.Size([1]), exchangeable='none'), RealSpace('observe37', torch.Size([1]), exchangeable='none'), RealSpace('observe29', torch.Size([1]), exchangeable='none'), RealSpace('observe36', torch.Size([1]), exchangeable='none'), RealSpace('observe23', torch.Size([1]), exchangeable='none'), RealSpace('sample18', torch.Size([1]), exchangeable='none'), RealSpace('sample31', torch.Size([1]), exchangeable='none'), RealSpace('sample19', torch.Size([1]), exchangeable='none'), RealSpace('sample20', torch.Size([1]), exchangeable='none'), RealSpace('sample6', torch.Size([1]), exchangeable='none'), RealSpace('sample35', torch.Size([1]), exchangeable='none'), RealSpace('observe22', torch.Size([1]), exchangeable='none'), RealSpace('sample13', torch.Size([1]), exchangeable='none'), RealSpace('sample14', torch.Size([1]), exchangeable='none'), RealSpace('observe9', torch.Size([1]), exchangeable='none'), RealSpace('sample4', torch.Size([1]), exchangeable='none'), RealSpace('sample2', torch.Size([1]), exchangeable='none'), RealSpace('sample1', torch.Size([1]), exchangeable='none'), RealSpace('sample33', torch.Size([1]), exchangeable='none'), RealSpace('sample26', torch.Size([1]), exchangeable='none'), RealSpace('observe30', torch.Size([1]), exchangeable='none'), RealSpace('observe8', torch.Size([1]), exchangeable='none'), RealSpace('sample32', torch.Size([1]), exchangeable='none'), RealSpace('sample3', torch.Size([1]), exchangeable='none'), RealSpace('sample21', torch.Size([1]), exchangeable='none'), RealSpace('sample11', torch.Size([1]), exchangeable='none'), RealSpace('observe16', torch.Size([1]), exchangeable='none')]

    def get_graphical_model_mask_and_indices(self):
        indices = [set([i]) for i in range(38)]
        indices[0].add(3)
        indices[3].add(0)
        indices[2].add(13)
        indices[13].add(2)
        indices[4].add(3)
        indices[3].add(4)
        indices[5].add(32)
        indices[32].add(5)
        indices[8].add(0)
        indices[0].add(8)
        indices[8].add(2)
        indices[2].add(8)
        indices[8].add(10)
        indices[10].add(8)
        indices[8].add(17)
        indices[17].add(8)
        indices[8].add(34)
        indices[34].add(8)
        indices[9].add(13)
        indices[13].add(9)
        indices[10].add(22)
        indices[22].add(10)
        indices[16].add(22)
        indices[22].add(16)
        indices[17].add(14)
        indices[14].add(17)
        indices[18].add(22)
        indices[22].add(18)
        indices[26].add(32)
        indices[32].add(26)
        indices[27].add(0)
        indices[0].add(27)
        indices[27].add(2)
        indices[2].add(27)
        indices[27].add(10)
        indices[10].add(27)
        indices[27].add(17)
        indices[17].add(27)
        indices[27].add(34)
        indices[34].add(27)
        indices[28].add(0)
        indices[0].add(28)
        indices[28].add(2)
        indices[2].add(28)
        indices[28].add(10)
        indices[10].add(28)
        indices[28].add(17)
        indices[17].add(28)
        indices[28].add(34)
        indices[34].add(28)
        indices[29].add(14)
        indices[14].add(29)
        indices[30].add(13)
        indices[13].add(30)
        indices[33].add(14)
        indices[14].add(33)
        indices[34].add(32)
        indices[32].add(34)
        indices[36].add(3)
        indices[3].add(36)
        indices_lists = [list(idx_set) for idx_set in indices]
        max_attendable = max([len(idx_list) for idx_list in indices_lists])
        mask = torch.zeros(38, max_attendable, dtype=torch.long)
        for [i, idx_list] in enumerate(indices_lists):
            mask[i][slice(None, len(idx_list))] = 1
        padded_indices = [idx_list + [0] * (max_attendable - len(idx_list)) for idx_list in indices_lists]
        indices_tensor = torch.tensor(padded_indices, dtype=torch.long)
        return [mask, indices_tensor]

    def run_simulator(self):
        sample34_val = Beta(8, 2).sample()
        sample12_val = Normal(1.0, 0.3).sample()
        sample5_val = Normal(1.0, 0.3).sample()
        sample27_val = Beta(8, 2).sample()
        sample28_val = Beta(2, 8).sample()
        sample0_val = Beta(2, 8).sample()
        sample25_val = Normal(2.5, 0.3).sample()
        sample7_val = Beta(2, 8).sample()
        observe37_val = Beta(10, 10).sample()
        observe23_val = Beta(10, 10).sample()
        sample18_val = Normal(2.5, 0.3).sample()
        sample19_val = Normal(1.0, 0.3).sample()
        sample20_val = Beta(8, 2).sample()
        sample6_val = Beta(8, 2).sample()
        sample35_val = Beta(2, 8).sample()
        sample13_val = Beta(8, 2).sample()
        sample14_val = Beta(2, 8).sample()
        observe9_val = Beta(10, 10).sample()
        sample4_val = Normal(2.5, 0.3).sample()
        sample2_val = Beta(1, 50).sample()
        sample1_val = Beta(1, 20).sample()
        sample33_val = Normal(1.0, 0.3).sample()
        sample26_val = Normal(1.0, 0.3).sample()
        observe30_val = Beta(10, 10).sample()
        sample32_val = Normal(2.5, 0.3).sample()
        sample21_val = Beta(2, 8).sample()
        sample11_val = Normal(2.5, 0.3).sample()
        observe16_val = Beta(10, 10).sample()
        sample10_val = Bernoulli(1.0 / (1.0 + torch.exp(-(5.0 * (0.4 * sample0_val + 0.4 * sample1_val + 0.2 * sample2_val + -0.2 + -0.4))))).sample()
        sample24_val = Bernoulli(1.0 / (1.0 + torch.exp(-(5.0 * (0.4 * sample0_val + 0.4 * sample1_val + 0.2 * sample2_val + 4.0 + -0.4))))).sample()
        sample17_val = Bernoulli(1.0 / (1.0 + torch.exp(-(5.0 * (0.4 * sample0_val + 0.4 * sample1_val + 0.2 * sample2_val + 0.2 + -0.4))))).sample()
        sample31_val = Bernoulli(1.0 / (1.0 + torch.exp(-(5.0 * (0.4 * sample0_val + 0.4 * sample1_val + 0.2 * sample2_val + 6.5 + -0.4))))).sample()
        sample3_val = Bernoulli(1.0 / (1.0 + torch.exp(-(5.0 * (0.4 * sample0_val + 0.4 * sample1_val + 0.2 * sample2_val + -0.5 + -0.4))))).sample()
        observe15_val = Normal(sample11_val if sample10_val else sample12_val, 0.2).sample()
        observe29_val = Normal(sample25_val if sample24_val else sample26_val, 0.2).sample()
        observe36_val = Normal(sample32_val if sample31_val else sample33_val, 0.2).sample()
        observe22_val = Normal(sample18_val if sample17_val else sample19_val, 0.2).sample()
        observe8_val = Normal(sample4_val if sample3_val else sample5_val, 0.2).sample()
        return [Sample(self.spaces[0], sample10_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[1], sample34_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[2], sample24_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[3], observe15_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[4], sample12_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[5], sample5_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[6], sample27_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[7], sample28_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[8], sample0_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[9], sample25_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[10], sample17_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[11], sample7_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[12], observe37_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[13], observe29_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[14], observe36_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[15], observe23_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[16], sample18_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[17], sample31_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[18], sample19_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[19], sample20_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[20], sample6_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[21], sample35_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[22], observe22_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[23], sample13_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[24], sample14_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[25], observe9_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[26], sample4_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[27], sample2_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[28], sample1_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[29], sample33_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[30], sample26_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[31], observe30_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[32], observe8_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED), Sample(self.spaces[33], sample32_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[34], sample3_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[35], sample21_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[36], sample11_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * LATENT), Sample(self.spaces[37], observe16_val.unsqueeze(0), torch.ones(1, dtype=torch.long) * OBSERVED)]

    def sample_states_and_edges(self, marg_prop):
        samples = self.run_simulator()
        states = OrderedDict([[s.space.id, s.value] for s in samples])
        translated_edges = marginal_edges(self.sparse_attention_and_indices, states)
        return [states, translated_edges]

    def __len__(self):
        return self.config.data.test_dataset_length if self.is_test else self.config.data.dataset_length

    def __getitem__(self, index, marginalize=True):
        samples = self.run_simulator()
        [states, translated_edges] = self.states_and_edges[index % self.config.data.n_states]
        return [flatten_samples(samples), translated_edges]

    def get_sample(self, index):
        """Get a sample with proper RNG seeding for reproducibility."""
        import numpy as np
        import torch
        np.random.seed(index)
        torch.manual_seed(index)
        samples = self.run_simulator()
        valid_states_and_edges = [self.sample_states_and_edges(marg_prop=0) for _ in range(self.config.data.n_states)]
        [states, translated_edges] = valid_states_and_edges[index % self.config.data.n_states]
        return [samples, flatten_samples(samples), translated_edges]

    def avg_log_prob(self, N=None):
        """Return average log probability (not used)."""
        return torch.tensor(0.0)

    def validation_metrics(self, samples, data, model):
        return {}

    def plot(self, samples, data, prefix):
        return None
