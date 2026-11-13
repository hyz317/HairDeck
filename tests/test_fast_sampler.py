"""Fast-sampler settings and weight-cache lifetime across model residency changes."""
from contextlib import nullcontext, redirect_stdout
import gc
import io
import os
import unittest
from unittest.mock import patch
import weakref

import torch

from hairdeck.infer import memory
from hairdeck.models.setdit import SetDiT
from hairdeck.sampling import fast


def model(device='cpu'):
    return SetDiT(4, dim=16, depth=1, heads=2, cond_dim=4, head_dim=4, n_ctx=2).to(device).eval()


def tensor_refs(weights):
    return [weakref.ref(t) for pair in weights.w.values() for t in pair if t is not None]


class CompileSettingsTests(unittest.TestCase):
    def test_compile_follows_fast_unless_explicitly_overridden(self):
        cases = [
            (None, {}, False),
            (None, {'HAIRDECK_FAST': '1'}, True),
            (None, {'HAIRDECK_FAST': '1', 'HAIRDECK_COMPILE': '0'}, False),
            (None, {'HAIRDECK_COMPILE': '1'}, True),
            ('7.2', {}, True),
            ('7.2', {'HAIRDECK_FAST': '0'}, False),
            ('7.2', {'HAIRDECK_FAST': '0', 'HAIRDECK_COMPILE': '1'}, True),
            ('7.2', {'HAIRDECK_COMPILE': '0'}, False),
        ]
        for hip, env, expected in cases:
            with self.subTest(hip=hip, env=env), patch.object(torch.version, 'hip', hip), \
                    patch.dict(os.environ, env, clear=True):
                self.assertEqual(fast._compile_on(), expected)


class CompileFallbackTests(unittest.TestCase):
    def test_creation_and_call_failures_fall_back_without_retaining_tensors(self):
        for stage in ('creation', 'call'):
            with self.subTest(stage=stage), patch.object(fast, '_COMPILE_FAILED', []):
                refs = []

                def fail(*args, **kwargs):
                    retained = torch.ones(4)
                    refs.append(weakref.ref(retained))
                    raise RuntimeError('compiler probe failed')

                def eager(value):
                    return value + 1

                options = {'side_effect': fail} if stage == 'creation' else {'return_value': fail}
                output = io.StringIO()
                with patch.object(torch, 'compile', **options), redirect_stdout(output):
                    compiled = fast._compiled(eager)
                    self.assertEqual(compiled(3), 4)
                    self.assertEqual(compiled(5), 6)
                self.assertEqual(len(refs), 1)  # a failed compiler is not retried
                self.assertEqual(output.getvalue().count('sampling eagerly'), 1)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))


class WeightCacheTests(unittest.TestCase):
    def test_resident_weights_reuse_cache_without_changing_state_dict(self):
        net = model()
        before = {name: tensor.clone() for name, tensor in net.state_dict().items()}
        cached = fast._weights(net)
        self.assertIs(fast._weights(net), cached)
        self.assertEqual(set(before), set(net.state_dict()))
        for name, tensor in net.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_offload_releases_cached_tensors_and_rebuilds_after_reload(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        if torch.backends.mps.is_available():
            devices.append('mps')
        for device in devices:
            with self.subTest(device=device):
                net = model(device)
                cached = fast._weights(net)
                held_dictionary = cached.w  # a denoiser can still reference this dictionary
                refs = tensor_refs(cached)
                self.assertTrue(refs)
                memory.move_model(net, 'cpu')
                self.assertTrue(all(p.device.type == 'cpu' for p in net.parameters()))
                self.assertFalse(held_dictionary)
                self.assertTrue(all(ref() is None for ref in refs))
                with torch.no_grad():
                    net.inp.weight.fill_(0.75)
                memory.move_model(net, device)
                rebuilt = fast._weights(net)
                self.assertIsNot(rebuilt, cached)
                torch.testing.assert_close(rebuilt.w[net.inp][0], net.inp.weight.to(torch.bfloat16), rtol=0, atol=0)
                self.assertEqual(rebuilt.w[net.inp][0].device.type, device)
                memory.move_model(net, 'cpu')

    def test_changed_parameters_release_old_copies_before_rebuilding(self):
        net = model()
        cached = fast._weights(net)
        refs = tensor_refs(cached)
        with torch.no_grad():
            net.inp.weight.add_(1)
        original = fast._Weights

        def rebuild(m):
            self.assertTrue(all(ref() is None for ref in refs))
            return original(m)

        with patch.object(fast, '_Weights', side_effect=rebuild):
            rebuilt = fast._weights(net)
        self.assertIsNot(rebuilt, cached)
        torch.testing.assert_close(rebuilt.w[net.inp][0], net.inp.weight.to(torch.bfloat16), rtol=0, atol=0)

    def test_failed_transfer_still_releases_cache(self):
        net = model()
        cached = fast._weights(net)
        refs = tensor_refs(cached)
        with patch.object(net, 'to', side_effect=RuntimeError('transfer failed')):
            with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
                memory.move_model(net, 'cpu')
        self.assertFalse(cached.w)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_residency_eviction_and_request_failure_release_cache(self):
        net = model()
        policy = memory.ModelResidency('cpu', mode='off')
        policy.register('generation', net)
        policy.register('prep', torch.nn.Linear(4, 4))
        policy.enabled = True  # exercise lifecycle with real CPU modules and no CUDA driver
        with patch.object(torch.cuda, 'device', side_effect=lambda device: nullcontext()), \
                patch.object(torch.cuda, 'empty_cache'), patch.object(policy, '_snapshot', return_value=({}, (0, 0))):
            cached = fast._weights(net)
            policy.activate('prep')
            self.assertFalse(cached.w)
            policy.activate('generation')
            cached = fast._weights(net)
            refs = tensor_refs(cached)
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                with policy.request():
                    raise RuntimeError('request failed')
            self.assertFalse(policy.on_gpu)
            self.assertTrue(all(ref() is None for ref in refs))


if __name__ == '__main__':
    unittest.main()
