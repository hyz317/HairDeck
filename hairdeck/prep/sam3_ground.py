"""SAM 3 text-prompt grounding of the preprocessing: the image model with its processor (confidence 0.5), the text
features of all prompts computed once, and one batched grounding pass of several prompts on an image.

SAM 3 (facebookresearch/sam3, SAM License) is imported on first use: either the installed sam3 package or a clone at
hairdeck.paths.sam3_repo(); the checkpoint is weights/sam3/sam3.pt (the gated facebook/sam3 download).
"""
import os
import sys

import torch

from .. import paths


def _sam3():
    try:
        import sam3  # noqa: F401
    except ImportError:
        repo = paths.sam3_repo()
        if not os.path.isdir(repo):
            raise ImportError('SAM 3 is not installed: pip install it from facebookresearch/sam3 or set HAIRDECK_SAM3_REPO '
                              'to a clone') from None
        sys.path.insert(0, repo)
    from sam3.model.data_misc import FindStage
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    return FindStage, Sam3Processor, build_sam3_image_model


def load_model(compile=False):
    _, Sam3Processor, build_sam3_image_model = _sam3()
    model = build_sam3_image_model(checkpoint_path=paths.weights('sam3', 'sam3.pt'), load_from_HF=False, compile=compile)
    return model, Sam3Processor(model, confidence_threshold=0.5)


@torch.inference_mode()
def text_feats(model, prompts):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model.backbone.forward_text(prompts, device='cuda')


@torch.inference_mode()
def ground(model, state, tf, idx):
    """Grounding for the prompts idx (indices into tf) on the image in state; returns probs (P, Q), boxes (P, Q, 4)
    cxcywh normalised, mask logits (P, Q, h, w)."""
    FindStage = _sam3()[0]
    n = len(idx)
    bo = dict(state['backbone_out'])
    bo['language_features'] = tf['language_features'][:, idx]
    bo['language_mask'] = tf['language_mask'][idx]
    if 'language_embeds' in tf:
        bo['language_embeds'] = tf['language_embeds'][:, idx] if tf['language_embeds'].dim() == 3 else tf['language_embeds']
    fs = FindStage(img_ids=torch.zeros(n, device='cuda', dtype=torch.long),
                   text_ids=torch.arange(n, device='cuda', dtype=torch.long),
                   input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None,
                   input_points_mask=None)
    out = model.forward_grounding(backbone_out=bo, find_input=fs, geometric_prompt=model._get_dummy_prompt(n),
                                  find_target=None)
    probs = (out['pred_logits'].sigmoid() * out['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
    return probs.float(), out['pred_boxes'].float(), out['pred_masks']
