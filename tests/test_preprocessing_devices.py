"""Device-routing regression tests, using CPU tensors and recorded CUDA targets.

"""
from contextlib import nullcontext
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from hairdeck import dev
from hairdeck.infer import image, memory
from hairdeck.prep import matte, preprocess, sam3_ground


class Sam3DeviceTests(unittest.TestCase):
    def test_processor_uses_compute_device_independently_of_weight_storage(self):
        for storage, compute, expected in [('cpu', 'cuda:1', 'cuda:1'),
                                           ('cuda:1', None, 'cuda:1'), ('cuda', None, 'cuda')]:
            with self.subTest(storage=storage, compute=compute):
                model = Mock()
                builder, processor = Mock(return_value=model), Mock()
                with patch.object(sam3_ground, '_sam3', return_value=(object, processor, builder)), \
                        patch.object(sam3_ground.paths, 'weights', return_value='sam3.pt'):
                    sam3_ground.load_model(device=storage, inference_device=compute)
                self.assertEqual(builder.call_args.kwargs['device'], storage)
                self.assertEqual(str(processor.call_args.kwargs['device']), expected)

    def test_text_features_follow_explicit_device(self):
        model = types.SimpleNamespace(backbone=Mock())
        with patch.object(sam3_ground.torch, 'autocast', return_value=nullcontext()):
            sam3_ground.text_feats(model, ['hair'], device=torch.device('cuda:1'))
        args = model.backbone.forward_text.call_args
        self.assertEqual(args.args[0], ['hair'])
        self.assertEqual(str(args.kwargs['device']), 'cuda:1')

    def test_grounding_indices_follow_compute_device(self):
        tf = dict(language_features=torch.zeros(2, 3, 4), language_mask=torch.zeros(3, 2),
                  language_embeds=torch.zeros(2, 3, 4))
        result = dict(pred_logits=torch.zeros(2, 1, 1), presence_logit_dec=torch.zeros(2, 1),
                      pred_boxes=torch.zeros(2, 1, 4), pred_masks=torch.zeros(2, 1, 2, 2))
        model = Mock()
        model.forward_grounding.return_value = result
        stage = Mock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
        allocations = []
        zeros, arange = torch.zeros, torch.arange

        def allocate(factory):
            def call(*args, **kwargs):
                allocations.append(str(kwargs.pop('device')))
                return factory(*args, **kwargs, device='cpu')
            return call

        with patch.object(sam3_ground, '_sam3', return_value=(stage, object, object)), \
                patch.object(sam3_ground.torch, 'zeros', side_effect=allocate(zeros)), \
                patch.object(sam3_ground.torch, 'arange', side_effect=allocate(arange)):
            probs, boxes, masks = sam3_ground.ground(model, {'backbone_out': {}}, tf, [0, 2], device='cuda:1')
        self.assertEqual(allocations, ['cuda:1', 'cuda:1'])
        torch.testing.assert_close(stage.call_args.kwargs['img_ids'], torch.tensor([0, 0]))
        torch.testing.assert_close(stage.call_args.kwargs['text_ids'], torch.tensor([0, 1]))
        self.assertEqual(probs.shape, (2, 1))
        self.assertIs(boxes, result['pred_boxes'])
        self.assertIs(masks, result['pred_masks'])


class PrepDeviceTests(unittest.TestCase):
    def test_prep_passes_compute_device_in_both_residency_modes(self):
        for mode in ('on', 'off'):
            with self.subTest(mode=mode):
                policy = memory.ModelResidency('cuda:1', mode=mode)
                with patch.object(preprocess, 'load_model', return_value=(Mock(), Mock())) as load, \
                        patch.object(preprocess, 'build_matte', return_value=types.SimpleNamespace(net=Mock())) as build:
                    prep = preprocess.Prep({}, matte='birefnet_matting', residency=policy)
                self.assertEqual(str(prep.device), 'cuda:1')
                expected_storage = 'cpu' if mode == 'on' else 'cuda:1'
                for call in (load.call_args, build.call_args):
                    self.assertEqual(str(call.kwargs['device']), expected_storage)
                    self.assertEqual(str(call.kwargs['inference_device']), 'cuda:1')

    def test_standalone_prep_keeps_default_device(self):
        expected = str(dev.select())  # cuda off macOS (hairdeck.dev.DEFAULT); favour (MPS) on macOS
        with patch.object(preprocess, 'load_model', return_value=(Mock(), Mock())) as load:
            prep = preprocess.Prep({})
        self.assertEqual(str(prep.device), expected)
        self.assertEqual(str(load.call_args.kwargs['device']), expected)
        self.assertEqual(str(load.call_args.kwargs['inference_device']), expected)

    def test_sam_stage_keeps_runtime_device_for_prompts_and_crop_canvas(self):
        policy = types.SimpleNamespace(device=torch.device('cuda:1'), initial_device='cpu',
                                       register=Mock(), activate=Mock())
        with patch.object(preprocess, 'load_model', return_value=(Mock(), Mock())):
            prep = preprocess.Prep({}, residency=policy)
        prompt = prep.prompts[0]
        ground_result = (torch.ones(1, 1), torch.zeros(1, 1, 4), torch.ones(1, 1, 2, 2))
        targets = []
        zeros = torch.zeros

        def canvas(*args, **kwargs):
            targets.append(str(kwargs.pop('device')))
            return zeros(*args, **kwargs, device='cpu')

        with patch.object(preprocess, 'text_feats', return_value={'features': 'cached'}) as text, \
                patch.object(preprocess, 'ground', return_value=ground_result) as ground, \
                patch.object(preprocess.torch, 'autocast', return_value=nullcontext()), \
                patch.object(preprocess.torch, 'zeros', side_effect=canvas):
            instances, _ = prep._sam(Image.new('RGB', (2, 2)), [prompt], (4, 4), paste=(1, 1, 2, 2))
        policy.activate.assert_called_once_with('sam3')
        self.assertEqual(str(text.call_args.kwargs['device']), 'cuda:1')
        self.assertEqual(str(ground.call_args.kwargs['device']), 'cuda:1')
        self.assertEqual(targets, ['cuda:1'])
        self.assertEqual(instances[prompt][0][1].sum(), 4)

    def test_alpha_postprocess_follows_returned_tensor_device(self):
        alpha = np.array([[255, 128], [0, 255]], dtype=np.uint8)
        rgba = np.full((2, 2, 4), 120, dtype=np.uint8)
        rgba[..., 3] = alpha
        returned = (torch.full((2, 2), 255, dtype=torch.uint8),
                    torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 2, dtype=torch.bool), {}, {})
        _, person, hair, face = preprocess.one(Mock(return_value=returned), Image.fromarray(rgba))
        np.testing.assert_array_equal(person, alpha)
        np.testing.assert_array_equal(hair, alpha > 127)
        np.testing.assert_array_equal(face, alpha > 127)

    def test_preprocessor_peak_is_for_selected_device(self):
        pre = preprocess.Preprocessor.__new__(preprocess.Preprocessor)
        pre.name = 'test'
        pre.prep = types.SimpleNamespace(device=torch.device('cuda:1'))
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        arrays = [np.ones((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)]
        with patch.object(preprocess, 'one', return_value=({'box': [0, 0, 2, 2], 'status': 'ok'}, *arrays)), \
                patch.object(preprocess.torch.cuda, 'max_memory_allocated', return_value=2**30) as peak:
            result = pre(rgb, image_path='unused.png')
        self.assertEqual(str(peak.call_args.args[0]), 'cuda:1')
        self.assertEqual(result['info']['matting']['peak_gpu_gb'], 1)

    def test_pipeline_synchronizes_requested_device(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        prep = dict(box=None, notes=[], rgb=rgb, hair=np.zeros((2, 2), dtype=bool),
                    status='no_hair_no_face', backend='test', info={})
        owner = image.Pipeline.__new__(image.Pipeline)
        owner.dev = 'cuda:1'
        owner.preprocessor = object()
        owner.prep = Mock(return_value=(prep, None))
        owner._finish = lambda result, started: result
        owner.residency = memory.ModelResidency('cuda:1', mode='off')
        for cache_enabled in (False, True):
            with self.subTest(cache_enabled=cache_enabled):
                owner.cache_enabled = cache_enabled
                owner.clear_image_cache()
                with patch.object(image, 'load_image', return_value=(rgb, None, None, 'unused.png')), \
                        patch.object(image, 'hair_color', return_value=[0, 0, 0]), \
                        patch.object(image, 'prep_panel', return_value=object()), \
                        patch.object(image.torch.cuda, 'synchronize') as sync:
                    result = image.Pipeline.run.__wrapped__(owner, 'unused.png', write=False)
                self.assertFalse(result['ok'])
                self.assertEqual(sync.call_count, 2)
                self.assertTrue(all(str(call.args[0]) == 'cuda:1' for call in sync.call_args_list))


class MattingDeviceTests(unittest.TestCase):
    def test_matting_compute_placement_and_cpu_numeric_path(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, value):
                self.inputs.append(value.detach().clone())
                return [value.mean(dim=1, keepdim=True)]

        net = Net()
        factory = types.SimpleNamespace(BiRefNet=lambda **kw: net)
        allocations, transfers = [], []
        tensor, to = torch.tensor, torch.Tensor.to

        def buffer(*args, **kwargs):
            allocations.append(str(kwargs.pop('device')))
            return tensor(*args, **kwargs, device='cpu')

        def transfer(value, *args, **kwargs):
            if args and str(args[0]) == 'cuda:1':
                transfers.append(str(args[0]))
                args = ('cpu', *args[1:])
            return to(value, *args, **kwargs)

        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 10
        with patch.dict(sys.modules), patch.object(matte.paths, 'weights', return_value='unused'), \
                patch.object(matte, '_load_module', side_effect=[None, factory]), \
                patch('safetensors.torch.load_file', return_value={}), \
                patch.object(matte.torch, 'tensor', side_effect=buffer):
            model = matte.build('birefnet_matting', 4, device='cpu', inference_device='cuda:1')
        with patch.object(torch.Tensor, 'to', new=transfer):
            output = model(rgb)
        self.assertEqual(str(model.device), 'cuda:1')
        self.assertEqual(allocations, ['cuda:1', 'cuda:1'])
        self.assertEqual(transfers, ['cuda:1'])
        self.assertEqual(net.inputs[0].dtype, torch.float16)
        x = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255,
                          (4, 4), mode='bilinear', align_corners=False)
        mean = tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        expected_input = ((x - mean) / std).half()
        expected = F.interpolate(expected_input.mean(dim=1, keepdim=True).sigmoid().float(),
                                 (2, 3), mode='bilinear', align_corners=False)[0, 0]
        torch.testing.assert_close(net.inputs[0], expected_input, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
