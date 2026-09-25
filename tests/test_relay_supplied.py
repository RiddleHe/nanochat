import unittest
from types import SimpleNamespace
import torch
from transformers import AttentionInterface, Qwen3Config, Qwen3ForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from scripts.inspect import qwen_relay_supplied_common as c
from scripts.inspect import qwen_entity_relay_fixed_window as relay
from scripts.inspect import qwen_relay_paper_runtime as w


def encoded(span=(1, 2)):
    return c.ab.EncodedPrompt(0, 'test', 0, 'two words', 'test', '', list(range(2, 9)),
                              list('abcdefg'), span[0], entity_positions=span)


class SuppliedTests(unittest.TestCase):
    def test_frozen_panel(self):
        entities, _ = c.ab.load_entities(c.DATA)
        self.assertEqual(len(entities), 100)
        self.assertEqual(c.TIDS, [0, 1, 2, 3, 7])
        self.assertEqual(c.ab.TEMPLATES[7].name, 'name_badge')

    def test_pair_validation(self):
        one, two = encoded((1,)), encoded()
        c.validate_pair(two, two)
        with self.assertRaises(ValueError): c.validate_pair(one, two)
        changed = encoded(); changed.ids[4] = 90
        with self.assertRaises(ValueError): c.validate_pair(two, changed)

    def test_postsoftmax_full_span_removal_no_renormalization(self):
        torch.manual_seed(21)
        q, k, v = [torch.randn(1, 2, 7, 4) for _ in range(3)]
        m = SimpleNamespace(layer_idx=0, num_key_value_groups=1, is_causal=True, training=False)
        causal = torch.ones(7, 7, dtype=torch.bool).tril()
        scores = (q @ k.transpose(-1, -2) / 2).masked_fill(~causal, -torch.inf)
        p = scores.softmax(-1)
        ordinary = p @ v
        for disabled in ((), (0,)):
            with c.policy(encoded(), disabled) as pol:
                got, _ = c.ab.entity_zero_attention_forward(m, q, k, v, None, scaling=0.5)
                expected_p = p.clone()
                expected_p[:, :, 3:6, 1:3] = 0
                if disabled: expected_p[:, :, 6, 1:3] = 0
                torch.testing.assert_close(got.transpose(1, 2), expected_p @ v, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(got[:, :3].transpose(1, 2), ordinary[:, :, :3], rtol=1e-5, atol=1e-6)
                self.assertEqual(pol.intermediate_applications, 3)
                self.assertTrue((expected_p[:, :, 3:6].sum(-1) < 1).all())

    def test_two_token_entity_and_identity(self):
        torch.manual_seed(19); torch.set_num_threads(1)
        cfg = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=48,
                          num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, max_position_embeddings=128, bos_token_id=0,
                          eos_token_id=1, pad_token_id=0, attention_dropout=0.)
        cfg._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(cfg).eval(); layers = list(model.model.layers)
        tok = SimpleNamespace(pad_token_id=0, eos_token_id=1, decode=lambda ids, **kw: str(ids))
        device = torch.device('cpu'); enc = encoded()
        hook = c.configure_model(model)
        AttentionInterface.register('sdpa', c.ab.entity_zero_attention_forward)
        try:
            clean = relay.capture_donor_entity_states(model, layers, enc, device)
            own, _ = c.capture(model, layers, enc, device)
            for a, b in zip(clean, own):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(a.shape, (2, 32))
            for s, t in [(0, 4), (16, 20), (31, 35)]:
                for disabled in ((), w.outside_layers(36, s, t)):
                    direct = c.generate(model, tok, layers, enc, device, disabled)
                    identity = c.run_relay(model, tok, layers, enc, device, own[s], s, t, disabled)
                    self.assertEqual(direct['generated_token_ids'], identity['generated_token_ids'])
            with self.assertRaises(ValueError):
                c.run_relay(model, tok, layers, enc, device, own[0][:1], 0, 4)
        finally:
            hook.remove(); AttentionInterface.register('sdpa', sdpa_attention_forward)


if __name__ == '__main__': unittest.main()
