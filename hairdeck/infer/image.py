"""One character image -> hair cards of the image model, spring-bone chains and GLB files in the model's head frame
(y up, face toward +z, metres).

Stages:
  prep    hairdeck.prep.Preprocessor (SAM 3 + BiRefNet-matting, preset "best"): person matte, hair and face masks, crop
          box; the matte composited on white and cropped to the model resolution (crop_white)
  dino    DINOv3 ViT-L/16 tokens of the crop
  dit     ImgSetDiT, Euler sampling with classifier-free guidance, started from the noise of the given seed
  decode  card-VAE decoding of the tokens to cards (hairdeck.cards.tokens.tokens_to_cards)
  colors  optional: the colour model (hairdeck.models.colormodel) on the clean card tokens, the crop and the person
          matte -> per card 16 colours along it (sRGB)
  chains  optional: the chain DiT on the card tokens, Euler 50, CFG 2
  glb     card mesh with per-card UVs, coloured with the colours of the colour model (without it: the hair colour of the
          input's hair mask), a grey head proxy (the canonical skull ellipsoid) and the chains (orange tubes) as
          separate meshes

Use:
  P = Pipeline()                                    # loads SAM 3, BiRefNet-matting, DINOv3, the image DiT, the card
  r = P.run(image, seed=0, steps=50, cfg=2.0)       #   VAE, the colour model and the chain DiT once; HAIRDECK_OFFLOAD
                                                    #   controls residency
  r['ok'], r['message'], r['prep_vis'], r['model_input'], r['timings'], r['out_dir'],
  r['samples'][k]: 'cards', 'n_cards', 'chains', 'n_chains', 'glb' (cards + head), 'glb_nohead', 'glb_chains',
  'glb_nohead_chains', 'json', 'chains_json'
Chains can also be added later to a saved result: add_chains(out_dir, seed).
Each run writes out_dir/: input.png, prep.png (preprocessing panel), model_input.png, cards_seed<s>.json (the cards and
their colours), tokens_seed<s>.npy (raw card tokens), chains_seed<s>.json, hair_seed<s>[_nohead][_chains].glb,
info.json.
"""
import json
import os
import time
import uuid

import numpy as np
import torch
from PIL import Image, ImageOps

from .. import dev, paths
from ..cards.mesh import cards_to_json
from ..ckpt import load as load_ckpt
from ..cards.tokens import LatentDecoder, clean_tokens, sign_token_normals, tokens_to_cards
from ..data.crop import apply_crop, composite
from ..data.illus import crop_white
from ..models.colormodel import lab_to_srgb, load_color_dit
from ..models.imgmodel import DinoEncoder, ImgSetDiT
from ..models.tokenizer import LatentTokenizer
from ..prep.preprocess import Preprocessor
from ..sampling.flow import fm_sample
from ..spring.chaindit import ChainDiT, chain_stats, decode as chain_decode
from .glb import chain_trimesh, hair_color, hair_trimesh, head_trimesh, write_glb
from .panel import prep_panel
from .memory import ModelResidency, managed_request

IMAGE_DIT = ('image_model', 'model.safetensors')  # under paths.ckpt(): the released image model
CHAIN_DIT = ('chain_model', 'model.safetensors')  # the released chain model (head frame)
COLOR_DIT = ('color_model', 'model.safetensors')  # the released colour model
# the files of one sample in out_dir: the key of its record in run()'s 'samples' -> the name (s = the seed)
SAMPLE_FILES = {'json': 'cards_seed{s}.json', 'tokens': 'tokens_seed{s}.npy', 'chains_json': 'chains_seed{s}.json',
                'glb': 'hair_seed{s}.glb', 'glb_nohead': 'hair_seed{s}_nohead.glb', 'glb_chains': 'hair_seed{s}_chains.glb',
                'glb_nohead_chains': 'hair_seed{s}_nohead_chains.glb'}
ZERO_GPU = bool(os.environ.get('SPACES_ZERO_GPU'))  # set by Hugging Face on ZeroGPU Spaces (the `spaces` package reads it)
MAX_SIDE = 4096  # larger inputs are downscaled first (Lanczos); SAM 3 itself works at 1008 px and the model at 448 px


class PipelineError(Exception):
    pass


def euler_cfg(model, ctx, x, steps, cfg):
    """hairdeck.sampling.flow.fm_sample(model, ctx, n, d, steps, cfg, shift=1.0, sampler='euler') started from the given
    noise x (fm_sample draws x = randn(B, n, d, generator) itself; the same operations otherwise)."""
    B = ctx.shape[0]
    from ..sampling import fast
    if fast.enabled(model):  # the same steps, restructured for small batches (sampling/fast.py)
        return fast.euler(model, ctx, x, steps, cfg)
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)

    def vel(x, t):
        tt = torch.full((B,), float(t), device=ctx.device)
        if cfg == 1.0:
            return model(x, tt, ctx).float()
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([ctx, ctx]),
                     torch.cat([torch.zeros(B, dtype=torch.bool, device=ctx.device),
                                torch.ones(B, dtype=torch.bool, device=ctx.device)])).float()
        vc, vu = both.chunk(2)
        return vu + cfg * (vc - vu)

    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        x = x + (tn - t) * vel(x, t)
    return x


def load_image(image, want_path=False):
    """path / PIL / array -> (rgb (H, W, 3) uint8, alpha (H, W) uint8 or None, note), plus with want_path the path of the
    file when the pixels are the file's own (not EXIF-rotated, not downscaled), else None."""
    src = None
    if isinstance(image, (str, os.PathLike)):
        src = str(image)
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        a = image
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        image = Image.fromarray(a)
    if image is None:
        raise PipelineError('No image received.')
    try:
        rotated = image.getexif().get(0x0112, 1) not in (1, None)
    except Exception:  # noqa: BLE001
        rotated = False
    image = ImageOps.exif_transpose(image)
    note = None
    if rotated:
        src, note = None, 'EXIF orientation applied'
    if max(image.size) > MAX_SIDE:
        s = MAX_SIDE / max(image.size)
        note = f'downscaled from {image.size[0]}x{image.size[1]}'
        image = image.resize((round(image.width * s), round(image.height * s)), Image.Resampling.LANCZOS)
        src = None
    # an alpha channel as hairdeck.prep.preprocess.load_image sees one
    has_a = image.mode in ('RGBA', 'LA', 'PA', 'RGBa', 'La') or 'transparency' in image.info
    if has_a:
        im = image.convert('RGBA')
        a = np.asarray(im)
        out = np.ascontiguousarray(a[..., :3]), np.ascontiguousarray(a[..., 3]), note
    else:
        out = np.asarray(image.convert('RGB')), None, note
    return out + (src,) if want_path else out


class Pipeline:
    def __init__(self, ckpt=None, chain_ckpt=None, device=None, preprocessor=None, chains=True, out_root=None,
                 offload=None, colors=True, color_ckpt=None):
        """ckpt / chain_ckpt / color_ckpt: the image DiT, the chain DiT and the colour model (default: IMAGE_DIT,
        CHAIN_DIT and COLOR_DIT under paths.ckpt()); preprocessor: any callable with Preprocessor's call and result
        (default: Preprocessor('best')); out_root: where run() puts a result without an out_dir (default
        results/image_to_hair). Without the chain DiT or the colour model (chains / colors False, or a file that does
        not load: chain_error / color_error) the pipeline runs without chains / with one hair colour;
        offload: auto/on/off, otherwise HAIRDECK_OFFLOAD (Windows auto, Linux off);
        device: otherwise HAIRDECK_DEVICE (hairdeck.dev.select: favour -> CUDA, else MPS, else CPU)."""
        self.dev = device = dev.select(device)
        self.out_root = out_root or paths.results('image_to_hair')
        t0 = time.time()
        self.residency = ModelResidency(device, offload, zero_gpu=ZERO_GPU)
        initial_device = self.residency.initial_device
        ckpt = ckpt or paths.ckpt(*IMAGE_DIT)
        self.preprocessor = preprocessor or Preprocessor('best', residency=self.residency)
        ck = load_ckpt(ckpt)
        a = argparse_ns(ck['args'])
        assert a.rep == 'rgb', a.rep
        self.a = a
        self.tok = LatentTokenizer(json.load(open(paths.data('token_stats_head.json'))),
                                   json.load(open(paths.ckpt(a.vae, 'latent_stats_head.json'))), device=self.dev)
        self.enc = DinoEncoder(a.dino, a.res).to(initial_device)
        self.model = ImgSetDiT(self.tok.dim, a.dim, a.depth, a.heads, enc_dim=self.enc.dim,
                               n_ctx=self.enc.n_tokens).to(initial_device).eval()
        self.model.load_state_dict(ck['ema'])
        del ck
        self.vae = LatentDecoder(paths.ckpt(a.vae), self.dev)
        self.head = head_trimesh()
        self.ckpt = ckpt
        self.color, self.color_error = None, None
        if colors:
            try:
                net, ca = load_color_dit(color_ckpt or paths.ckpt(*COLOR_DIT), self.tok.dim, initial_device)
                # the same card tokens and model input
                assert (ca.vae, ca.dino, ca.res) == (a.vae, a.dino, a.res), (ca.vae, ca.dino, ca.res)
                self.color = net
            except Exception as e:  # the pipeline works with one hair colour
                self.color_error = f'{type(e).__name__}: {e}'
        self.chain, self.chain_error = None, None
        if chains:
            try:
                cc = load_ckpt(chain_ckpt or paths.ckpt(*CHAIN_DIT))
                ca = argparse_ns(cc['args'])
                assert ca.vae == a.vae, (ca.vae, a.vae)  # the same card token space
                net = ChainDiT(cc['dim'], ca.dim, ca.depth, ca.heads).to(initial_device).eval()
                net.load_state_dict(cc['ema'])
                self.chain_dim, self.chain_stats, self.chain_decode = cc['dim'], chain_stats(), chain_decode
                self.chain = net  # set last: a failure above leaves chains off instead of half loaded
            except Exception as e:  # the pipeline works without chains
                self.chain_error = f'{type(e).__name__}: {e}'
        self.residency.register('generation', self.enc, self.model, self.vae.vae.model, self.color, self.chain)
        if not ZERO_GPU:  # ZeroGPU: models are placed on cuda while no real GPU is attached (CUDA emulation)
            dev.sync(self.dev)
        self.load_sec = round(time.time() - t0, 1)

    # ------------------------------------------------------------ stages
    def prep(self, rgb, alpha=None, image_path=None, preprocessor=None):
        """the preprocessing; then the matte composited on white (an empty matte keeps the whole image) and cropped to
        the model resolution (crop_white). Returns (prep dict, model-input crop or None)."""
        p = (preprocessor or self.preprocessor)(rgb, alpha, image_path)
        if p['box'] is None:
            return p, None
        alpha_m = p['person'].copy()
        if alpha_m.max() <= 127:  # an empty matte keeps the whole image
            alpha_m[:] = 255
        full = np.asarray(composite(Image.fromarray(np.concatenate([p['rgb'], alpha_m[..., None]], -1), 'RGBA')))
        return p, crop_white(full, tuple(p['box']), self.a.res)

    def noise(self, seed):
        """The initial noise of fm_sample for one image, from a generator seeded `seed`."""
        g = dev.generator(self.dev, seed)
        return dev.randn((1, self.a.nmax, self.tok.dim), self.dev, g)

    def sample(self, crop, x0, steps, cfg):
        """-> (cards, raw card tokens (1, 192, 40))"""
        self.residency.activate('generation')
        img = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(self.dev)
        with torch.no_grad(), dev.amp(self.dev):
            ctx = self.model.encode_cond(self.enc(img))
            x = euler_cfg(self.model, ctx, x0, steps, cfg)
        return tokens_to_cards(self.tok, x, self.vae, fix=True)[0], x

    def matte_crop(self, p):
        """The person matte of a preprocessing result cut to the model input (bool (res, res)); an empty one: all true
        (as prep keeps the whole image for an empty matte)."""
        m = np.asarray(apply_crop(Image.fromarray(p['person']), tuple(p['box']), self.a.res,
                                  resample=Image.Resampling.NEAREST)) > 127
        return m if m.any() else np.ones_like(m)

    def color_context(self, crop, matte):
        """The colour model's context for one model input (crop (res, res, 3) uint8, matte bool (res, res))."""
        self.residency.activate('generation')
        img = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(self.dev).float()
        with torch.no_grad(), dev.amp(self.dev):
            return self.color.encode_ctx(self.enc(img), img, torch.from_numpy(matte)[None].to(self.dev))

    def card_colors(self, ctx, x, n_cards):
        """Colours of generated cards: per card (in the order of tokens_to_cards) 16 sRGB colours along it, from root to
        tip (hairdeck.cards.mesh.card_vertex_colors), for raw card tokens x and a colour context (color_context);
        rounded to the 4 decimals of the cards file, so the GLB files are the same from the cards file (add_chains)."""
        self.residency.activate('generation')
        xc = sign_token_normals(self.tok, clean_tokens(self.tok, x.float()))
        slots = torch.nonzero(xc[0, :, self.tok.groups['exist']][:, 0] > 0)[:, 0]
        if len(slots) != n_cards:
            raise PipelineError(f'{len(slots)} card slots for {n_cards} cards')
        with torch.no_grad(), dev.amp(self.dev):
            lab = self.color(xc, ctx)
        return np.round(lab_to_srgb(lab[0].float()[slots].cpu().numpy()), 4)

    def chains(self, x, seed, cfg=2.0, steps=50):
        """Spring-bone chains for generated card tokens x: the chain DiT, Euler 50, CFG 2, its own generator seeded with
        the card seed, head frame -> list of dict(path, joints, n_joints, stiffness)."""
        self.residency.activate('generation')
        g = dev.generator(self.dev, seed)
        with torch.no_grad(), dev.amp(self.dev):
            ctx = self.chain.encode_cond(clean_tokens(self.tok, x.float()))
            xc = fm_sample(self.chain, ctx, 48, self.chain_dim, steps, cfg, 1.0, g, 'euler')
        return self.chain_decode(xc[0].float().cpu().numpy(), self.chain_stats)

    # ------------------------------------------------------------ one request
    @managed_request
    def add_chains(self, out_dir, seed):
        """Chains for a saved result that was generated without them (tokens_seed{s}.npy, cards_seed{s}.json and
        info.json in out_dir): writes chains_seed{s}.json and the two *_chains.glb files, updates info.json and returns
        that sample's record."""
        if self.chain is None:
            raise PipelineError(f'the chain DiT is not loaded: {self.chain_error}')
        info = json.load(open(f'{out_dir}/info.json'))
        smp = next(s for s in info['samples'] if s['seed'] == seed)
        if smp.get('glb_chains'):
            return smp
        x = torch.from_numpy(np.load(f'{out_dir}/tokens_seed{seed}.npy')).to(self.dev)
        cj = json.load(open(f'{out_dir}/cards_seed{seed}.json'))
        cards = [(np.asarray(c['seq'], np.float64), 1 if c['shape'] == 'triangle' else 0) for c in cj]
        card_srgb = [c['colors'] for c in cj] if cj and all('colors' in c for c in cj) else None
        chains = self.chains(x, seed)
        smp['n_chains'] = len(chains)
        smp['chains_json'] = f'{out_dir}/chains_seed{seed}.json'
        json.dump(chains, open(smp['chains_json'], 'w'))
        hm, cm = hair_trimesh(cards, info['hair_color'], card_srgb=card_srgb), chain_trimesh(chains)
        if hm is not None and cm is not None:
            smp['glb_chains'] = f'{out_dir}/hair_seed{seed}_chains.glb'
            smp['glb_nohead_chains'] = f'{out_dir}/hair_seed{seed}_nohead_chains.glb'
            write_glb(smp['glb_chains'], hm, self.head, cm)
            write_glb(smp['glb_nohead_chains'], hm, None, cm)
        json.dump(info, open(f'{out_dir}/info.json.tmp', 'w'), indent=1, default=str)
        os.replace(f'{out_dir}/info.json.tmp', f'{out_dir}/info.json')
        return smp

    @managed_request
    def run(self, image, seed=0, steps=50, cfg=2.0, n_samples=1, out_dir=None, x0=None, write=True, chains=True, prep=None,
            colors=True):
        """image: path, PIL image or (H, W, 3|4) uint8 array; n_samples variations with the seeds seed, seed + 1, ... (at
        most 4); out_dir (default <out_root>/<time>_<id>) gets the files listed in the module docstring;
        prep: another preprocessor for this call; colors False: one hair colour instead of the colour model's"""
        T = {}
        sync = lambda: dev.sync(self.dev)
        t_all = time.time()
        rgb, alpha, size_note, src = load_image(image, want_path=True)
        backend = prep or self.preprocessor
        sync()
        t0 = time.time()
        p, crop = self.prep(rgb, alpha, src, backend)
        sync()
        T['prep'] = time.time() - t0
        notes = list(p['notes']) + ([size_note] if size_note else [])
        col = hair_color(p['rgb'], p['hair'])
        if out_dir is None and write:
            out_dir = os.path.join(self.out_root, f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}')
        if write:
            os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(np.concatenate([rgb, alpha[..., None]], -1) if alpha is not None else rgb).save(
                f'{out_dir}/input.png', compress_level=1)
        vis = prep_panel(p, crop)
        res = dict(ok=False, message='', prep_vis=vis, model_input=Image.fromarray(crop) if crop is not None else None,
                   samples=[], timings=T, out_dir=out_dir, status=p['status'], notes=notes, box=p['box'],
                   hair_color=col, prep_backend=p['backend'], prep_info=p['info'],
                   params=dict(seed=int(seed), steps=int(steps), cfg=float(cfg), n_samples=int(n_samples),
                               chains=bool(chains), colors=bool(colors)))
        if write:
            vis.save(f'{out_dir}/prep.png')
        seeds = [int(seed) + k for k in range(max(1, min(4, int(n_samples))))]
        if crop is None:
            res['message'] = ("Could not find the character's hair or face in this image, so no head crop could be made. "
                              "Please try an image in which the head is clearly visible.")
            if write:  # the files of these seeds from an earlier run of out_dir are not this request's
                for f in (f'{out_dir}/{n.format(s=s)}' for s in seeds for n in SAMPLE_FILES.values()):
                    if os.path.exists(f):
                        os.remove(f)
            return self._finish(res, t_all)
        if write:
            Image.fromarray(crop).save(f'{out_dir}/model_input.png')
        T['dino+dit+decode'] = 0.0
        want_chains = bool(chains) and self.chain is not None
        cctx = None
        if colors and self.color_error:
            res['notes'].append(f'one hair colour: the colour model is not loaded ({self.color_error})')
        if colors and self.color is not None:
            sync()
            t0 = time.time()
            cctx = self.color_context(crop, self.matte_crop(p))
            sync()
            T['colors'] = time.time() - t0
        if want_chains:
            T['chains'] = 0.0
        T['glb'] = 0.0
        for k, s in enumerate(seeds):
            sync()
            t0 = time.time()
            x = (x0 if (x0 is not None and k == 0) else self.noise(s))
            cards, tokens = self.sample(crop, x, int(steps), float(cfg))
            sync()
            t1 = time.time()
            card_srgb = None
            if cctx is not None and cards:
                try:
                    card_srgb = self.card_colors(cctx, tokens, len(cards))
                except Exception as e:
                    res['notes'].append(f'colours failed: {type(e).__name__}: {e}')
                sync()
                T['colors'] += time.time() - t1
                t1 = time.time()
            chains = None
            if want_chains and cards:
                try:
                    chains = self.chains(tokens, s)
                except Exception as e:
                    res['notes'].append(f'chains failed: {type(e).__name__}: {e}')
            sync()
            t2 = time.time()
            smp = dict(seed=s, cards=cards, n_cards=len(cards), n_meshed=int(sum(len(q) >= 3 for q, _ in cards)),
                       n_chains=None if chains is None else len(chains), colors=card_srgb is not None, glb=None,
                       glb_nohead=None, glb_chains=None, glb_nohead_chains=None, json=None, chains_json=None, tokens=None)
            hm = hair_trimesh(cards, col, card_srgb=card_srgb) if cards else None
            cm = chain_trimesh(chains) if chains else None
            if write:
                smp['json'] = f'{out_dir}/cards_seed{s}.json'
                json.dump(cards_to_json(cards, card_srgb), open(smp['json'], 'w'))
                if tokens is not None:  # raw card tokens, for chains added later
                    smp['tokens'] = f'{out_dir}/tokens_seed{s}.npy'
                    np.save(smp['tokens'], tokens.float().cpu().numpy())
                if chains is not None:
                    smp['chains_json'] = f'{out_dir}/chains_seed{s}.json'
                    json.dump(chains, open(smp['chains_json'], 'w'))
                if hm is not None:
                    smp['glb'] = f'{out_dir}/hair_seed{s}.glb'
                    smp['glb_nohead'] = f'{out_dir}/hair_seed{s}_nohead.glb'
                    write_glb(smp['glb'], hm, self.head)
                    write_glb(smp['glb_nohead'], hm)
                    if cm is not None:
                        smp['glb_chains'] = f'{out_dir}/hair_seed{s}_chains.glb'
                        smp['glb_nohead_chains'] = f'{out_dir}/hair_seed{s}_nohead_chains.glb'
                        write_glb(smp['glb_chains'], hm, self.head, cm)
                        write_glb(smp['glb_nohead_chains'], hm, None, cm)
                written = {smp[k] for k in SAMPLE_FILES if smp[k]}
                for f in (f'{out_dir}/{n.format(s=s)}' for n in SAMPLE_FILES.values()):
                    if f not in written and os.path.exists(f):  # of an earlier run of this seed, e.g. chains before --no-chains
                        os.remove(f)
            smp['chains'] = chains
            T['dino+dit+decode'] += t1 - t0
            if want_chains:
                T['chains'] += t2 - t1
            T['glb'] += time.time() - t2
            res['samples'].append(smp)
        empty = [s['seed'] for s in res['samples'] if s['n_meshed'] == 0]
        if len(empty) == len(seeds):
            res['message'] = 'No hair cards were generated for this image. Please try another seed or another image.'
        else:
            res['ok'] = True
            if empty:
                res['message'] = f'No hair cards were generated for seed {", ".join(map(str, empty))}.'
        return self._finish(res, t_all)

    def _finish(self, res, t_all):
        res['memory'] = self.residency.report()
        T = res['timings']
        T['total'] = time.time() - t_all
        res['timings'] = {k: round(v, 3) for k, v in T.items()}
        if res['out_dir']:
            info = {k: v for k, v in res.items() if k not in ('prep_vis', 'model_input', 'samples')}
            info['samples'] = [{k: v for k, v in s.items() if k not in ('cards', 'chains')} for s in res['samples']]
            info['ckpt'] = self.ckpt
            json.dump(info, open(f'{res["out_dir"]}/info.json', 'w'), indent=1, default=str)
        return res



def argparse_ns(d):
    import argparse
    return argparse.Namespace(**d)
