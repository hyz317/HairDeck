"""Gradio demo: upload one character image, get 3D hair cards in a viewer and as GLB files, spring-bone chains, a head-turn
preview and a VRM 0.x avatar with the hair (hairdeck.infer.image and hairdeck.vrm). Output paths, model names and the
preprocessing details stay in the server log and each result's info.json, not on the page.
python demo/app.py [--port 7860] [--no-warmup]
Local runs bind to 127.0.0.1 (on a server: ssh -N -L PORT:127.0.0.1:PORT HOST); on a Hugging Face Space (SPACE_ID set)
0.0.0.0, with the GPU calls through `spaces` on ZeroGPU. Results go to HAIRDECK_DEMO_OUT (default results/demo), the
example images are the list HAIRDECK_DEMO_EXAMPLES (default data/demo_examples/examples.json, paths relative to the
workspace root).
"""
import argparse
import contextlib
import html
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
if not os.environ.get('SPACE_ID'):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
import gradio as gr  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hairdeck import paths  # noqa: E402
from hairdeck.infer import image as PL  # noqa: E402
from hairdeck.vrm import export as VX  # noqa: E402

try:  # ZeroGPU (Hugging Face Spaces): a GPU only inside @spaces.GPU functions; the decorator has no effect elsewhere
    import spaces  # noqa: E402
    GPU = spaces.GPU
except ImportError:  # without `spaces` (a local run): plain calls
    def GPU(fn=None, **kw):
        return fn if callable(fn) else (lambda f: f)

OUT_ROOT = os.environ.get('HAIRDECK_DEMO_OUT') or paths.results('demo')
MAXS = 4
TITLE = 'HairDeck: Animation-Ready Stylized Hair Cards from a Single Image'  # the paper's title


def _examples():
    """The example images: a json list of image paths (relative to the workspace root, or absolute), HAIRDECK_DEMO_EXAMPLES
    or data/demo_examples/examples.json; none if the list is missing."""
    f = paths.resolve(os.environ.get('HAIRDECK_DEMO_EXAMPLES') or os.path.join('data', 'demo_examples', 'examples.json'))
    if not os.path.exists(f):
        return []
    return [p for p in (paths.resolve(x) for x in json.load(open(f))) if os.path.exists(p)]


EXAMPLES = _examples()
P = None
PORT = 7860
# one local GPU: in-process GPU work (generate, chains) from different events never overlaps. ZeroGPU: every
# @spaces.GPU call runs in a worker process with a GPU of its own, so calls of different visitors run side by side
# (the platform queues them when the Space has no free GPU slot)
GPU_LOCK = contextlib.nullcontext() if PL.ZERO_GPU else threading.Lock()
GPU_CONCURRENCY = 4 if PL.ZERO_GPU else 1


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sample_label(seed, n_cards, n_chains=None):
    return f'Seed {seed} · {n_cards} cards' + (f' · {n_chains} chains' if n_chains is not None else '')


def views(state, show_head, show_chains):
    out = []
    for k in range(MAXS):
        if state and k < len(state):
            s = state[k]
            key = ('glb' if show_head else 'glb_nohead') + ('_chains' if show_chains and s.get('glb_chains') else '')
            out.append(gr.update(value=s[key], visible=True, label=s['label']))
        else:
            out.append(gr.update(value=None, visible=(k == 0), label='3D hair cards'))
    return out


# preprocessing notes that matter to a user (the others stay in info.json)
NOTES = {'matte_uncertain': 'The segmentation of this image may be inaccurate.',
         'box_from_low_scores': 'The hair or face was detected with low confidence, so the crop may be off.',
         'no_hair': 'No hair was detected; the crop is based on the face.',
         'no_person_mask': 'No person was detected; the whole image was used.'}
STATUS_EMPTY = '<div class="status-line"><span class="status-dot"></span><span>Add an image to get started.</span></div>'


def describe(r):
    """the status line under the viewers (Markdown), from a pipeline result or a saved info.json"""
    lines = []
    if r['message']:
        lines.append(f"<span class=\"warn\">{html.escape(r['message'])}</span>")
    done = [s for s in r['samples'] if s.get('glb')]
    if done:
        n = len(done)
        lines.append('<div class="status-line"><span class="status-dot ready"></span>'
                     f'<strong>{n} variation{"s" if n > 1 else ""} ready</strong>'
                     f'<span class="status-time">{r["timings"].get("total", 0):.1f} s</span></div>')
    notes = [NOTES[n] for n in dict.fromkeys(list(r['notes']) + [r['status']]) if n in NOTES]
    if notes:
        lines.append('<span class="muted">' + ' '.join(notes) + '</span>')
    return '\n\n'.join(lines) or STATUS_EMPTY


def outputs_of(r, prep_vis, show_head, show_chains):
    state = [dict(glb=s['glb'], glb_nohead=s['glb_nohead'], glb_chains=s.get('glb_chains'),
                  glb_nohead_chains=s.get('glb_nohead_chains'), out_dir=r['out_dir'], seed=s['seed'], n_cards=s['n_cards'],
                  json=s.get('json'), chains_json=s.get('chains_json'),
                  label=sample_label(s['seed'], s['n_cards'], s.get('n_chains')))
             for s in r['samples'] if s['glb']]
    # the viewers get their files in a second step (.then(views, ...)): a gr.Model3D that becomes visible together with
    # its value stayed blank in a test, so first only the visibility changes
    vis = [gr.update(value=None, visible=(k == 0 or k < len(state))) for k in range(MAXS)]
    choices = [(f"Seed {s['seed']}", s['seed']) for s in state]
    return [prep_vis, *vis, files_of(state), describe(r), state,
            gr.update(choices=choices, value=choices[0][1] if choices else None), None, None, None, TURN_HINT, VRM_HINT]


def files_of(state):
    keys = ('glb', 'glb_nohead', 'glb_chains', 'json', 'chains_json', 'vrm', 'turn')
    return [s[k] for s in state for k in keys if s.get(k)] or None


def _run_seconds(image, seed, steps, cfg, n_samples, chains):
    """ZeroGPU time limit of one generate call: a few seconds per variation (preprocessing, Euler 50, chains, GLBs), with
    a wide margin for the first call after a cold start"""
    return 30 + 10 * max(1, min(MAXS, int(n_samples)))


@GPU(duration=_run_seconds)
def gpu_run(image, seed, steps, cfg, n_samples, chains):
    return P.run(image, seed=seed, steps=steps, cfg=cfg, n_samples=n_samples, chains=chains)


@GPU(duration=30)
def gpu_add_chains(out_dir, seed):
    """returns the sample record; the caller updates its state (a ZeroGPU call runs in another process)"""
    return P.add_chains(out_dir, seed)


def generate(image, seed, steps, cfg, n_samples, show_head, show_chains):
    if image is None:
        raise gr.Error('Please upload an image first.')
    t0 = time.time()
    try:
        with GPU_LOCK:
            r = gpu_run(image, int(seed), int(steps), float(cfg), int(n_samples), bool(show_chains))
    except PL.PipelineError as e:
        raise gr.Error(str(e))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log('OOM', traceback.format_exc())
        raise gr.Error('Not enough memory to finish this request. Try fewer variations or a smaller image.')
    except Exception:
        log('ERROR', traceback.format_exc())
        raise gr.Error('Something went wrong while generating. Please try again or use another image.')
    log('REQUEST', r['out_dir'], r['status'], r['prep_backend'], r['notes'], [s['n_cards'] for s in r['samples']],
        r['timings'], 'wall %.2f' % (time.time() - t0))
    if r['message'] and not r['ok']:
        gr.Warning(r['message'])
    return outputs_of(r, r['prep_vis'], show_head, show_chains)


def toggle(state, show_head, show_chains):
    """checkbox change: switch GLB files; chains that were not generated yet are added now (chain DiT, ~0.6 s each)"""
    if show_chains and P.chain is not None:
        for s in state or []:
            if s.get('glb_chains') or not s.get('out_dir'):
                continue
            try:
                ensure_chains(s)
            except Exception:
                log('ERROR chains', traceback.format_exc())
                gr.Warning('Spring-bone chains could not be generated for this sample.')
    elif show_chains:
        log('chains requested but the chain DiT is not loaded:', P.chain_error)
        gr.Warning('Spring-bone chains are not available right now.')
    return [*views(state, show_head, show_chains), files_of(state or []), state]


def ensure_chains(s):
    """chains for one sample of the state (chain DiT, ~0.7 s) if it has none yet; updates the sample's entry"""
    if s.get('glb_chains') and os.path.exists(f"{s['out_dir']}/chains_seed{s['seed']}.json"):
        return
    if P.chain is None:
        log('chains requested but the chain DiT is not loaded:', P.chain_error)
        raise gr.Error('Spring-bone chains are not available right now.')
    with GPU_LOCK:
        smp = gpu_add_chains(s['out_dir'], s['seed'])
    s.update(glb_chains=smp.get('glb_chains'), glb_nohead_chains=smp.get('glb_nohead_chains'), chains_json=smp.get('chains_json'),
             label=sample_label(s['seed'], s['n_cards'], smp['n_chains']))
    log('CHAINS', s['out_dir'], s['seed'], smp['n_chains'])


def pick_sample(state, seed):
    for s in state or []:
        if seed is not None and int(s['seed']) == int(seed):
            return s
    return (state or [None])[0]


def vrm_text(seed, info):
    check = 'The file passed validation.' if info['ok'] else 'The file did not pass every validation check.'
    return (f"Seed {seed} · VRM 0.x · {info['size_mb']} MB. {check} "
            'Includes a sample avatar with the generated hair and spring bones.')


def turn_text(seed):
    return f"Seed {seed} · A 3-second head turn with spring-bone motion."


def make_vrm(state, seed):
    """VRM 0.x of one sample on the avatar body (hairdeck.vrm.export.export_vrm; chains are computed if missing)"""
    s = pick_sample(state, seed)
    if s is None:
        raise gr.Error('Please generate hair first.')
    t0 = time.time()
    try:
        ensure_chains(s)
        d = s['out_dir']
        vi = f"{d}/vrm_seed{s['seed']}/vrm_info.json"
        if os.path.exists(VX.vrm_path(d, s['seed'])) and os.path.exists(vi):
            info = json.load(open(vi))
        else:
            info = VX.export_vrm(d, s['seed'], json.load(open(f'{d}/info.json'))['hair_color'])
    except gr.Error:
        raise
    except Exception:
        log('ERROR vrm', traceback.format_exc())
        raise gr.Error('The VRM export failed. Please try again.')
    s['vrm'] = info['vrm']
    tm = info['timings']
    text = vrm_text(s['seed'], info)
    log('VRM', d, s['seed'], info['ok'], info['problems'][:3], tm, 'wall %.1f' % (time.time() - t0))
    return [info['vrm'], info.get('rest_png'), text, files_of(state), state]


def make_turn(state, seed, show_chains):
    """3 s head-turn video of one sample with its spring chains simulated (hairdeck.vrm.export.turn_preview)"""
    s = pick_sample(state, seed)
    if s is None:
        raise gr.Error('Please generate hair first.')
    t0 = time.time()
    try:
        ensure_chains(s)
        d = s['out_dir']
        res = VX.turn_preview(d, s['seed'], json.load(open(f'{d}/info.json'))['hair_color'], show_chains=bool(show_chains))
    except gr.Error:
        raise
    except Exception:
        log('ERROR turn', traceback.format_exc())
        raise gr.Error('The head-turn preview failed. Please try again.')
    s['turn'] = res['out']
    text = turn_text(s['seed'])
    log('TURN', d, s['seed'], res.get('sec'), 'wall %.1f' % (time.time() - t0))
    return [res['out'], text, files_of(state), state]


def restore(show_head, show_chains, state, request: gr.Request):
    """?result=<output directory name>: show a saved result again (its info.json, prep.png and GLBs)"""
    rid = request.query_params.get('result') if request is not None else None
    if not rid:
        return [gr.update()] * (MAXS + 3) + [state] + [gr.update()] * 6
    d = os.path.join(OUT_ROOT, os.path.basename(rid))
    if not os.path.exists(f'{d}/info.json'):
        gr.Warning('This result was not found.')
        return [gr.update()] * (MAXS + 3) + [state] + [gr.update()] * 6
    r = json.load(open(f'{d}/info.json'))
    out = outputs_of(r, f'{d}/prep.png' if os.path.exists(f'{d}/prep.png') else None, show_head, show_chains)
    st = out[MAXS + 3]
    if st:  # a turn video or VRM made earlier for the first sample is shown again
        s0 = st[0]
        for f in (f"{d}/turn_seed{s0['seed']}.mp4", f"{d}/turn_seed{s0['seed']}_chains.mp4"):
            if os.path.exists(f):
                out[MAXS + 5], s0['turn'] = f, f
                out[MAXS + 8] = turn_text(s0['seed'])
        if os.path.exists(VX.vrm_path(d, s0['seed'])):
            out[MAXS + 6] = s0['vrm'] = VX.vrm_path(d, s0['seed'])
            vi = f"{d}/vrm_seed{s0['seed']}/vrm_info.json"
            if os.path.exists(vi):
                out[MAXS + 9] = vrm_text(s0['seed'], json.load(open(vi)))
            if os.path.exists(f"{d}/vrm_seed{s0['seed']}/rest.png"):
                out[MAXS + 7] = f"{d}/vrm_seed{s0['seed']}/rest.png"
        for s in st:
            if os.path.exists(f"{d}/chains_seed{s['seed']}.json"):
                s['chains_json'] = f"{d}/chains_seed{s['seed']}.json"
        out[MAXS + 1] = files_of(st)
    return out


HEADER = f'''<header id="hdr">
  <h1>{TITLE}</h1>
  <nav class="project-links" aria-label="Project resources">
    <a href="https://hyzcluster.github.io/HairDeck" target="_blank" rel="noopener noreferrer">Project Page</a>
    <a href="https://huggingface.co/hyz317/HairDeck" target="_blank" rel="noopener noreferrer">Model</a>
    <a href="https://github.com/hyz317/HairDeck" target="_blank" rel="noopener noreferrer">Code</a>
    <a href="https://arxiv.org/abs/2610.40599" target="_blank" rel="noopener noreferrer">Paper</a>
  </nav>
</header>'''
FOOTER = '<footer id="ftr">For best results, use a clear portrait with the full hairstyle visible.</footer>'
CSS = """
.gradio-container {max-width: 1360px !important; margin: 0 auto !important; padding: 0 !important;}
.main.fill_width {padding: 0 !important;}
#app-shell {gap: 12px; padding: 14px 24px 10px; border: 0; background: transparent;}
#hdr {padding: 0;}
#hdr h1 {margin: 0; font-size: clamp(18px, 1.7vw, 22px); line-height: 1.25; font-weight: 700; color: #292136;}
#hdr .project-links {display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 8px;}
#hdr .project-links a, #hdr .project-links .is-pending {display: inline-flex; align-items: center; gap: 5px; min-height: 28px; padding: 3px 9px; border: 1px solid #e4deed; border-radius: 8px; background: #fff; color: #765098; font-size: 12px; line-height: 1.4; font-weight: 600; text-decoration: none;}
#hdr .project-links a:hover {background: #f3edf9; border-color: #cdb8e4;}
#hdr .project-links .is-pending {color: #9c91a7; background: #faf8fd;}
#hdr .project-links .soon {font-size: 10px; font-weight: 400;}
#workspace {display: grid !important; grid-template-columns: 320px minmax(0, 1fr); gap: 16px !important; align-items: start;}
#workspace > .column {min-width: 0 !important; width: 100%;}
#input-panel {gap: 10px; border: 1px solid #e4deed; border-radius: 16px; background: #fff; padding: 14px; box-shadow: 0 3px 14px #47315f05;}
#input-panel .form {background: transparent !important; border: 0 !important;}
#result-panel {gap: 12px;}
.section-heading {display: flex; align-items: center; gap: 10px;}
.section-heading .step {display: grid; place-items: center; width: 27px; height: 27px; background: #eee7f8; border-radius: 8px; color: #7a5a9e; font-size: 11px; font-weight: 700;}
.section-heading h2 {font-size: 16px; font-weight: 700; color: #352b43; margin: 0;}
.section-copy {font-size: 12px; line-height: 1.45; color: #80768c; margin: 3px 0 0;}
#input-image {border: 1px dashed #d3c5e5 !important; border-radius: 11px !important; background: #faf8fd; overflow: hidden;}
#input-image .upload-container {color: #85709f; background: #faf8fd;}
#input-image [data-testid="upload-text"] {font-size: 13px; line-height: 1.7; color: #85768f;}
#input-image [data-testid="upload-icon"] {color: #9471bb;}
#input-image .image-container {background: #faf8fd;}
#gen-btn {min-height: 46px; border-radius: 10px; font-size: 14px; font-weight: 700; box-shadow: none;}
#gen-btn:hover {box-shadow: 0 4px 12px #805aaf20;}
#examples-panel {border: 0; background: transparent; padding: 0;}
#examples-panel .label {font-size: 11px; color: #80758e;}
#examples-panel .label svg {display: none;}
#examples-panel .gallery:not(.container) {display: grid !important; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 6px; padding: 0;}
#examples-panel .gallery-item {min-width: 0; width: 100%; height: 54px; border: 1px solid #e6dfee; border-radius: 8px; overflow: hidden; padding: 0;}
#examples-panel .gallery-item .container {height: 100%; width: 100%; border: 0; padding: 0;}
#examples-panel .gallery-item img {width: 100% !important; height: 54px !important; object-fit: cover; object-position: center 20%;}
#examples-panel .gallery-item:hover {border-color: #9b7fc0;}
#sample-count {border: 0; padding: 0; background: transparent;}
#sample-count .wrap {display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;}
#sample-count label {justify-content: center; text-align: center; min-height: 34px; margin: 0; border-radius: 8px;}
#sample-count input {position: absolute; opacity: 0; width: 1px;}
#advanced {border: 0; border-top: 1px solid #f0ebf5; border-radius: 0; background: transparent;}
#advanced > .label-wrap {padding: 8px 0 0;}
#advanced .form {border: 0;}
#preview-panel {padding: 14px; border: 1px solid #e4deed; border-radius: 16px; background: white; gap: 8px; box-shadow: 0 3px 14px #47315f05;}
#preview-top {display: grid !important; grid-template-columns: minmax(200px,1fr) auto; align-items: center; gap: 16px;}
#preview-top > .block {flex: 1 1 auto !important; min-width: 160px !important;}
#viewer-toolbar {align-items: center; gap: 14px; justify-content: flex-end; min-width: 0; flex: 0 0 auto !important; width: auto !important;}
#viewer-toolbar > .block {min-width: 0 !important; flex: 0 0 auto !important; width: auto !important; background: transparent; padding: 0 !important; border: 0 !important;}
#viewer-toolbar .checkbox-container {border: 0; background: transparent; box-shadow: none; padding: 4px 0;}
#viewer-toolbar label {font-size: 12px; color: #766987;}
#preview-heading .section-copy {margin: 2px 0 0 37px; font-size: 11px;}
#model-grid {display: grid !important; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 12px !important;}
#model-grid > .model-viewer {min-width: 0 !important; width: 100% !important; height: 400px !important; border: 1px solid #e6dfed !important; border-radius: 12px !important; background: #f7f5fb !important; overflow: hidden !important;}
#model-grid:has(> .model-viewer:nth-child(2)) > .model-viewer {height: 300px !important;}
.model-viewer > label {margin: 11px !important; border: 1px solid #e8e0ef !important; background: #ffffffec !important; color: #6c5c7d !important; border-radius: 7px !important; padding: 5px 8px !important; font-size: 11px !important; box-shadow: none !important;}
.model-viewer > label svg {display: none;}
.model-viewer .empty {background: radial-gradient(ellipse at 50% 45%, #fff 0%, #f7f5fb 70%); flex-direction: column; gap: 20px; color: #9b8aa9;}
.model-viewer .empty .icon {display: none;}
.model-viewer .empty::before {content: ''; display: block; width: 54px; height: 72px; border: 1.5px solid #b8a2cf; border-radius: 8px; transform: rotate(-10deg); box-shadow: 9px 3px 0 -1px #f7f5fb, 10px 3px 0 0 #d2c2e5, -8px 4px 0 -1px #f7f5fb, -9px 4px 0 0 #e1d6ee;}
.model-viewer .empty::after {content: 'Your 3D hair will appear here'; font-size: 13px; color: #9783a9;}
#status {background: transparent; border: 0; padding: 1px 2px; min-height: 25px; font-size: 12px; color: #80708f;}
#status p {margin: 4px 0; line-height: 1.5;}
.status-line {display: flex; align-items: center; gap: 8px; min-height: 22px;}
.status-dot {height: 6px; width: 6px; flex: 0 0 6px; border-radius: 50%; background: #d0c5dc;}
.status-dot.ready {background: #9672bf;}
.status-line strong {font-weight: 700; color: #6b5185;}
.status-time {margin-left: auto; font-variant-numeric: tabular-nums; color: #9a8ca7;}
#status .warn {display: block; color: #936426; padding: 4px 0;}
.muted {color: #887c93;}
#result-tabs {border: 1px solid #e4deed; border-radius: 14px; background: white; padding: 3px 16px 12px;}
#result-tabs .tab-wrapper {border-bottom: 1px solid #eee7f5;}
#result-tabs .tab-container {gap: 18px; padding: 0 2px; border-bottom: 0;}
#result-tabs .tab-container button {padding: 13px 1px 11px; font-size: 12px; font-weight: 400; border-radius: 0; color: #8a7c96;}
#result-tabs .tab-container button.selected {color: #7b56a2; border-bottom-color: #a485c6;}
#result-tabs .tabitem {padding: 10px 0 2px; border: 0;}
.legend {font-size: 12px; line-height: 1.55; color: #82748f;}
.legend p {margin: 3px 0 8px;}
#asset-files {border: 1px solid #e9e2f1; border-radius: 10px; background: #fcfafd; min-height: 85px; max-height: 210px; overflow: auto;}
#asset-files .empty {min-height: 85px;}
#asset-files td:has(button[aria-label="Remove this file"]) {display: none;}
#sample-picker {border: 0; padding: 0; background: transparent !important; box-shadow: none;}
#sample-picker [data-testid="block-info"] {display: block; margin: 0 0 6px; padding: 0; font-size: 11px; line-height: 16px; font-weight: 700; color: #80658f;}
#sample-picker > .container > .wrap {height: 44px; min-height: 44px; border: 1px solid #d5c4e6; border-radius: 9px; background: #fff; box-shadow: none;}
#sample-picker > .container > .wrap:focus-within {outline: 2px solid #9471bb; outline-offset: 2px;}
#sample-picker input:focus-visible {outline: none !important;}
#sample-picker .wrap-inner {min-height: 42px; padding: 0 12px; align-items: center;}
#sample-picker .secondary-wrap {align-items: center;}
#lower-row {display: grid !important; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 12px !important; align-items: start;}
#lower-row > * {min-width: 0 !important; width: 100%;}
#animation-panel {padding: 14px; gap: 10px; border: 1px solid #d9c9eb; border-radius: 14px; background: #f3edf9;}
#animation-panel .section-heading .step {background: #e5d8f3; color: #79549e;}
#animation-panel .section-copy {margin: 2px 0 0 37px;}
#animation-actions {display: grid !important; grid-template-columns: minmax(120px,180px) repeat(2,minmax(120px,1fr)); align-items: end; gap: 10px;}
#animation-actions > * {min-width: 0 !important; width: 100%;}
#animation-actions > .form {background: transparent; border: 0; box-shadow: none;}
#animation-actions button {min-height: 44px; border-radius: 9px; font-size: 13px; font-weight: 700;}
#animation-actions button.secondary {background: #fff; border-color: #cdb8e4; color: #765098;}
#animation-previews {border: 0; border-top: 1px solid #e1d5ed; border-radius: 0; background: transparent;}
#animation-previews > .label-wrap {padding: 10px 0 0; color: #80658f; font-size: 12px;}
#animation-results .block {border-radius: 10px;}
#ftr {font-size: 11px; color: #a094aa; padding: 0 2px;}
:focus-visible {outline: 2px solid #9471bb !important; outline-offset: 3px;}
@media (max-width: 1200px) {
 #lower-row {grid-template-columns: minmax(0,1fr);}
}
@media (max-width: 1050px) {
 #app-shell {padding: 12px 16px 10px;}
 #workspace {grid-template-columns: 280px minmax(0,1fr); gap: 16px !important;}
 #input-panel {padding: 16px;}
 #preview-top {grid-template-columns: minmax(0,1fr); gap: 8px;}
 #viewer-toolbar {justify-content: flex-start;}
}
@media (max-width: 760px) {
 #app-shell {padding: 12px 12px;}
 #workspace {grid-template-columns: minmax(0,1fr); gap: 18px !important;}
 #app-shell {gap: 12px;}
 #input-image {height: 245px !important;}
 #input-panel, #preview-panel {padding: 16px; border-radius: 13px;}
 #model-grid > .model-viewer {height: 360px !important;}
 #model-grid:has(> .model-viewer:nth-child(2)) > .model-viewer {height: 310px !important;}
 #viewer-toolbar {justify-content: flex-start; flex-wrap: wrap; gap: 14px;}
 #preview-top {display: grid !important; grid-template-columns: minmax(0,1fr); gap: 10px !important;}
 #preview-top > * {min-width: 0 !important; width: 100% !important;}
 #animation-panel {padding: 16px;}
 #animation-actions {grid-template-columns: repeat(2,minmax(0,1fr));}
 #animation-actions > .form {grid-column: 1 / -1;}
 #result-tabs {padding-left: 12px; padding-right: 12px;}
 #result-tabs .tab-container {gap: 8px;}
 #result-tabs .tab-container button {font-size: 11px; padding: 12px 2px 10px;}
}
@media (prefers-reduced-motion: reduce) { * {transition: none !important; scroll-behavior: auto !important;} }
"""
PREP_LEGEND = ('The detected hair, face, and crop are shown on the left. The prepared portrait is shown on the right.')
FILES_LEGEND = ('Download the GLB meshes and geometry data for each variation. '
                'Motion previews and VRM avatars are included here after you create them.')
ANIM_LEGEND = ('Choose a variation to preview its movement or export it on a sample avatar.')
TURN_HINT = 'Click Preview motion to render a 3-second head turn of this variation.'
VRM_HINT = 'Click Export VRM to put this hair on a sample avatar.'
TURN_BUSY = 'Rendering the head turn. This can take up to a minute.'
VRM_BUSY = 'Exporting the VRM avatar. This can take up to a minute.'


def theme():
    """A quiet, offline workbench; all fonts and assets are local."""
    sans = ['Calibri', 'Segoe UI', 'Arial', 'sans-serif']
    return gr.themes.Base(primary_hue='purple', secondary_hue='purple', neutral_hue='slate', radius_size='lg',
                          font=sans, font_mono=['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace']).set(
        body_background_fill='#f7f5fb', body_background_fill_dark='#f7f5fb',
        body_text_color='#382d45', body_text_color_dark='#382d45',
        body_text_color_subdued='#897c95', body_text_color_subdued_dark='#897c95',
        background_fill_primary='#ffffff', background_fill_primary_dark='#ffffff',
        background_fill_secondary='#faf7fd', background_fill_secondary_dark='#faf7fd',
        panel_background_fill='transparent', panel_background_fill_dark='transparent',
        panel_border_width='0px', panel_border_width_dark='0px',
        border_color_primary='#e5ddee', border_color_primary_dark='#e5ddee',
        block_background_fill='#ffffff', block_background_fill_dark='#ffffff',
        block_border_color='#e6deef', block_border_color_dark='#e6deef',
        block_label_background_fill='#ffffff', block_label_background_fill_dark='#ffffff',
        block_label_text_color='#82718f', block_label_text_color_dark='#82718f',
        block_label_text_size='12px', block_label_shadow='none',
        block_title_text_color='#655376', block_title_text_color_dark='#655376',
        input_background_fill='#faf8fd', input_background_fill_dark='#faf8fd',
        input_border_color='#e3d9ed', input_border_color_dark='#e3d9ed', input_radius='8px',
        input_text_size='13px', input_shadow='none', input_shadow_dark='none',
        checkbox_label_background_fill='#faf7fd', checkbox_label_background_fill_dark='#faf7fd',
        checkbox_label_background_fill_selected='#eee5f8', checkbox_label_background_fill_selected_dark='#eee5f8',
        checkbox_label_text_color='#78648b', checkbox_label_text_color_dark='#78648b',
        checkbox_label_text_color_selected='#7d59a3', checkbox_label_text_color_selected_dark='#7d59a3',
        button_primary_background_fill='#8260ac', button_primary_background_fill_dark='#8260ac',
        button_primary_background_fill_hover='#704b9b', button_primary_background_fill_hover_dark='#704b9b',
        button_primary_border_color='#8260ac', button_primary_border_color_dark='#8260ac',
        button_primary_text_color='#ffffff', button_primary_text_color_dark='#ffffff',
        button_secondary_background_fill='#f0e9f8', button_secondary_background_fill_dark='#f0e9f8',
        button_secondary_text_color='#755292', button_secondary_text_color_dark='#755292',
        button_secondary_border_color='#dfd0ee', button_secondary_border_color_dark='#dfd0ee',
        button_primary_shadow='none', button_primary_shadow_dark='none',
        button_secondary_shadow='none', button_secondary_shadow_dark='none',
        slider_color='#a083c0', slider_color_dark='#a083c0', color_accent='#8260ac',
        layout_gap='12px', block_padding='12px', block_radius='10px')


def actions_available(state):
    return [gr.update(interactive=bool(state)), gr.update(interactive=bool(state))]


def clear_sample_previews():
    """User selection changes must not leave an export from a different sample visible."""
    return [None, None, None, TURN_HINT, VRM_HINT]


def turn_started():
    """the previews open at once with a line saying the work has started (the head turn takes up to a minute)"""
    return [gr.update(open=True), TURN_BUSY]


def vrm_started():
    return [gr.update(open=True), VRM_BUSY]


def preview_panel_state(state=None, seed=None):
    """Open saved/generated media, but keep an empty preview area compact."""
    sample = pick_sample(state, seed)
    return gr.update(open=bool(sample and (sample.get('turn') or sample.get('vrm'))))


def restore_source(request: gr.Request):
    """Restore the portrait alongside a saved result without changing the public output contract."""
    rid = request.query_params.get('result') if request is not None else None
    if rid:
        source = os.path.join(OUT_ROOT, os.path.basename(rid), 'input.png')
        if os.path.isfile(source):
            return gr.update(value=source)
    return gr.update()


def restore_settings(request: gr.Request):
    """Keep visible generation controls consistent with a reopened result."""
    rid = request.query_params.get('result') if request is not None else None
    if not rid:
        return [gr.update()] * 4
    path = os.path.join(OUT_ROOT, os.path.basename(rid), 'info.json')
    if not os.path.isfile(path):
        return [gr.update()] * 4
    params = json.load(open(path)).get('params', {})
    checks = [('seed', lambda v: isinstance(v, int) and v >= 0),
              ('steps', lambda v: v in (10, 50)),
              ('cfg', lambda v: isinstance(v, (int, float)) and 1 <= v <= 4),
              ('n_samples', lambda v: v in (1, 2, 3, 4))]
    return [gr.update(value=params[k]) if k in params and valid(params[k]) else gr.update()
            for k, valid in checks]


def build_ui():
    # The model-facing inputs and output order are deliberately unchanged.
    with gr.Blocks(title=TITLE, delete_cache=(6 * 3600, 2 * 86400), fill_width=True) as demo:
        state = gr.State([])
        with gr.Column(elem_id='app-shell'):
            gr.HTML(HEADER, js_on_load=None)
            with gr.Row(equal_height=False, elem_id='workspace'):
                with gr.Column(scale=3, min_width=280, elem_id='input-panel'):
                    gr.HTML('<div class="section-heading"><span class="step">01</span><h2>Source image</h2></div>'
                            '<p class="section-copy">Upload an image or choose an example.</p>')
                    image = gr.Image(type='filepath', image_mode=None, sources=['upload', 'clipboard'],
                                     label='Source image', show_label=False, height=230, elem_id='input-image',
                                     placeholder='Drop an image here or click to upload')
                    if EXAMPLES:
                        with gr.Column(elem_id='examples-panel', min_width=0):
                            gr.Examples(EXAMPLES, inputs=image, label='Try an example', examples_per_page=12)
                    n_samples = gr.Radio([1, 2, 3, 4], value=1, label='Variations', elem_id='sample-count')
                    with gr.Accordion('Advanced settings', open=False, elem_id='advanced'):
                        seed = gr.Number(value=0, precision=0, label='Seed', minimum=0,
                                         info='Change the seed for a different result.')
                        steps = gr.Radio([50, 10], value=50, label='Sampling steps',
                                         info='10 is faster. 50 uses more refinement steps.')
                        cfg = gr.Slider(1.0, 4.0, value=2.0, step=.25, label='Guidance scale')
                    btn = gr.Button('Generate hair', variant='primary', size='lg', elem_id='gen-btn')
                with gr.Column(scale=7, min_width=0, elem_id='result-panel'):
                    with gr.Column(elem_id='preview-panel'):
                        with gr.Row(elem_id='preview-top'):
                            gr.HTML('<div id="preview-heading"><div class="section-heading"><span class="step">02</span>'
                                    '<h2>3D preview</h2></div><p class="section-copy">Drag to orbit · Scroll to zoom</p></div>')
                            with gr.Row(elem_id='viewer-toolbar'):
                                show_head = gr.Checkbox(value=True, label='Reference head', container=False)
                                show_chains = gr.Checkbox(value=False, label='Spring bones', container=False)
                        with gr.Row(elem_id='model-grid'):
                            models = [gr.Model3D(label='3D hair cards', height=400, visible=(k == 0), interactive=False,
                                               clear_color=(.969, .961, .984, 1), elem_classes='model-viewer', min_width=0)
                                      for k in range(MAXS)]
                        info = gr.Markdown(STATUS_EMPTY, elem_id='status', sanitize_html=False)
                    with gr.Row(equal_height=False, elem_id='lower-row'):
                        with gr.Column(elem_id='animation-panel', min_width=0):
                            gr.HTML('<div class="section-heading"><span class="step">03</span>'
                                    '<h2>Animation &amp; VRM</h2></div><p class="section-copy">' + ANIM_LEGEND + '</p>')
                            with gr.Row(equal_height=False, elem_id='animation-actions'):
                                pick = gr.Dropdown(choices=[], label='Variation', scale=2, min_width=130,
                                                   elem_id='sample-picker')
                                turn_btn = gr.Button('Preview motion', variant='primary', scale=1, interactive=False)
                                vrm_btn = gr.Button('Export VRM', scale=1, interactive=False)
                            with gr.Accordion('Motion and avatar previews', open=False, elem_id='animation-previews') as previews:
                                with gr.Row(equal_height=False, elem_id='animation-results'):
                                    with gr.Column(min_width=220):
                                        turn_video = gr.Video(label='Motion preview', height=280, autoplay=True, loop=True,
                                                              interactive=False)
                                        turn_info = gr.Markdown(TURN_HINT, elem_classes='legend')
                                    with gr.Column(min_width=220):
                                        vrm_img = gr.Image(label='Avatar preview', type='filepath', interactive=False, height=280)
                                        vrm_file = gr.File(label='VRM avatar', interactive=False)
                                        vrm_info = gr.Markdown(VRM_HINT, elem_classes='legend')
                        with gr.Column(elem_id='downloads-col', min_width=0):
                            with gr.Tabs(elem_id='result-tabs', selected='export'):
                                with gr.Tab('Downloads', id='export'):
                                    gr.Markdown(FILES_LEGEND, elem_classes='legend')
                                    files = gr.File(label='Download assets', show_label=False, file_count='multiple',
                                                    interactive=False, elem_id='asset-files')
                                with gr.Tab('Input details', id='input-details'):
                                    prep_vis = gr.Image(label='Input processing', show_label=False, type='pil', format='png',
                                                        interactive=False, height=280)
                                    gr.Markdown(PREP_LEGEND, elem_classes='legend')
            gr.HTML(FOOTER, js_on_load=None)
        outs = [prep_vis, *models, files, info, state, pick, turn_video, vrm_file, vrm_img, turn_info, vrm_info]
        btn.click(generate, [image, seed, steps, cfg, n_samples, show_head, show_chains], outs,
                  concurrency_limit=GPU_CONCURRENCY, concurrency_id='gpu', api_name='generate').then(
                      views, [state, show_head, show_chains], models).then(
                          actions_available, state, [turn_btn, vrm_btn], queue=False, api_visibility='private')
        btn.click(preview_panel_state, [], previews, queue=False, api_visibility='private')
        show_head.change(views, [state, show_head, show_chains], models)
        show_chains.change(toggle, [state, show_head, show_chains], [*models, files, state],
                           concurrency_limit=GPU_CONCURRENCY, concurrency_id='gpu')
        vrm_btn.click(vrm_started, [], [previews, vrm_info], queue=False, api_visibility='private')
        ev = vrm_btn.click(make_vrm, [state, pick], [vrm_file, vrm_img, vrm_info, files, state], concurrency_limit=1,
                           concurrency_id='vrm', api_name='make_vrm')
        ev.success(preview_panel_state, [state, pick], previews, queue=False, api_visibility='private')
        ev.failure(lambda: VRM_HINT, [], vrm_info, queue=False, api_visibility='private')
        turn_btn.click(turn_started, [], [previews, turn_info], queue=False, api_visibility='private')
        ev = turn_btn.click(make_turn, [state, pick, show_chains], [turn_video, turn_info, files, state], concurrency_limit=1,
                            concurrency_id='turn', api_name='make_turn')
        ev.success(preview_panel_state, [state, pick], previews, queue=False, api_visibility='private')
        ev.failure(lambda: TURN_HINT, [], turn_info, queue=False, api_visibility='private')
        pick.input(clear_sample_previews, [], [turn_video, vrm_file, vrm_img, turn_info, vrm_info],
                   queue=False, api_visibility='private').then(
                       preview_panel_state, [], previews, queue=False, api_visibility='private')
        demo.load(restore, [show_head, show_chains, state], outs).then(
            views, [state, show_head, show_chains], models).then(
                actions_available, state, [turn_btn, vrm_btn], queue=False, api_visibility='private').then(
                    preview_panel_state, [state, pick], previews, queue=False, api_visibility='private')
        demo.load(restore_source, [], image, queue=False, api_visibility='private')
        demo.load(restore_settings, [], [seed, steps, cfg, n_samples], queue=False, api_visibility='private')
    return demo


def main():
    global P, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=int(os.environ.get('GRADIO_SERVER_PORT', '7860')))
    ap.add_argument('--no-warmup', action='store_true')
    a = ap.parse_args()
    PORT = a.port
    log('loading pipeline, GPU', os.environ.get('CUDA_VISIBLE_DEVICES'))
    P = PL.Pipeline(out_root=OUT_ROOT)
    log('loaded in', P.load_sec, 's; GPU memory', '-' if PL.ZERO_GPU else round(torch.cuda.memory_allocated() / 2 ** 30, 2),
        'GB; chains', 'on' if P.chain is not None else f'off ({P.chain_error})')
    if not a.no_warmup and EXAMPLES and not PL.ZERO_GPU:  # on ZeroGPU there is no GPU outside the @spaces.GPU calls
        r = P.run(EXAMPLES[0], seed=0, steps=10, cfg=2.0, write=False)
        log('warm-up', r['status'], [s['n_cards'] for s in r['samples']], r['timings'])
    try:
        VX.start_worker()  # Blender worker for the VRM export (builds the body's cache the first time, ~30 s)
        log('VRM worker started, log', VX.work_dir('worker.log'))
    except Exception as e:
        log('VRM worker not started (the export falls back to one Blender run per request):', e)
    demo = build_ui()
    demo.queue(default_concurrency_limit=8, max_size=64)  # generate / chains / turn / VRM set their own limits
    demo.launch(**launch_kwargs(a.port))


def english_ui_js():
    """Use this installed Gradio build's own locale switch, entirely offline.

    Gradio 6 selects built-in component translations from the browser locale;
    I18n dictionaries alone do not set that locale. Resolve the bundled export
    instead of hard-coding an asset hash or changing browser language settings.
    """
    assets = Path(gr.__file__).parent / 'templates' / 'frontend' / 'assets'
    core = next((p for p in sorted(assets.glob('core-*.js')) if 'changeLocale' in p.read_text()), None)
    finish = ("document.documentElement.lang = 'en'; "
              "document.querySelector('#input-image button.upload-container')?.setAttribute('aria-label', 'Upload an image');")
    if core is None:
        # The layout and explicit English copy must still work after SDK changes.
        # Gradio's built-in strings may then follow the browser's locale.
        return '() => {' + finish + '}'
    return ("async () => {if (!window.__hairDemoEnglish) {window.__hairDemoEnglish = (async () => {"
            "try {const {setupi18n, changeLocale} = await import(new URL("
            + json.dumps('assets/' + core.name)
            + ", document.baseURI).href); await setupi18n({}, 'en'); return {changeLocale};} "
            + "catch {console.warn('English component translations could not be initialized.');}})();} "
            + "const locale = await window.__hairDemoEnglish; locale?.changeLocale('en'); " + finish + '}')


def launch_kwargs(port):
    english = english_ui_js()
    return dict(server_name='0.0.0.0' if os.environ.get('SPACE_ID') else '127.0.0.1',
                server_port=port, share=False, show_error=True, ssr_mode=False,
                mcp_server=False, enable_monitoring=False, pwa=False, footer_links=[], max_file_size='60mb',
                allowed_paths=[OUT_ROOT] + sorted({os.path.dirname(e) for e in EXAMPLES}),
                theme=theme(), css=CSS, js=english, head='<script type="module">(' + english + ')();</script>')


if __name__ == '__main__':
    main()
