"""VRM 0.x export of a generated hairstyle on an avatar body, and the head-turn preview.

export_vrm(out_dir, seed, colour_srgb, body=None, render=True): a saved result of hairdeck.infer.image (out_dir/
cards_seed{s}.json and chains_seed{s}.json, head frame) -> out_dir/<file_tag>_seed{s}.vrm (vrm_path), the generated hair
and chains on the body (hairdeck.vrm.body; default data/avatar/body.json) with the body's own hair removed:
  1. hairdeck.vrm.prep (in this process): cards and chain joints into the body's dataset frame (p = (p_head - b) / k,
     width and thickness / k), mesh, UVs and skinning in the Blender world of the body; hair colour sRGB -> linear;
  2. Blender 4.2 with the VRM Add-on for Blender: blender/vrm_export.py (one Blender run), or the resident worker
     blender/vrm_worker.py when one runs for the body (start_worker()): removal of the body's hair, the new hair mesh,
     chain bones and VRM 0.x spring-bone groups (secondaryAnimation), export;
  3. hairdeck.vrm.check on the file; hairdeck.vrm.render rest-pose renders (front / side / back) in a subprocess.
Files in out_dir/vrm_seed{s}/ (intermediate names: hairdeck.vrm.body.FILES): run/ (cards and chains in the dataset
frame), prep/ (the prep npz and json), the gen VRM, the export json (what the Blender step did), the orig VRM (a link to
the add-on's unchanged re-export of the body, made by the worker; the reference of the round-trip checks), check.json,
rest.png, vrm_info.json (the returned dict) and the logs of the steps.
turn_preview(out_dir, seed, colour_srgb, show_chains=False, body=None): hairdeck.vis.turn_preview in a subprocess ->
out_dir/turn_seed{s}.mp4 (turn_seed{s}_chains.mp4 with the chain joints drawn) and its json.

Environment:
  HAIRDECK_BLENDER     command that runs Blender 4.2 LTS with the VRM Add-on for Blender (split like a shell command
                       line; default: bash ROOT/third_party/blender/bl.sh)
  HAIRDECK_GL          OpenGL platform of the render subprocesses: egl (default; EGL device = the first of
                       CUDA_VISIBLE_DEVICES) or osmesa (CPU)
  HAIRDECK_PYTHONPATH  PYTHONPATH of the subprocesses (e.g. vendored packages such as pyrender); otherwise none
  HAIRDECK_VRM_WORK    directory of the worker's job files (jobs/), its cache (ref/) and its log (default
                       results/vrm_export)

python -m hairdeck.vrm.export OUT_DIR SEED [vrm|turn|both] [--body BODY_JSON]   (hair colour from OUT_DIR/info.json)
"""
import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys
import time
import uuid

from .. import paths
from ..infer.glb import srgb_to_linear
from . import body as B
from .check import check
from .prep import prepare

RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(RELEASE, 'blender')
BLENDER = ('third_party', 'blender', 'bl.sh')  # default Blender wrapper, under paths.root()
WORK = 'vrm_export'  # default worker folder, under paths.results()


def blender_cmd():
    """Command prefix that runs Blender: HAIRDECK_BLENDER, else `bash ROOT/third_party/blender/bl.sh` (a wrapper that
    keeps Blender's user, cache and temporary paths inside the workspace)."""
    c = os.environ.get('HAIRDECK_BLENDER')
    return shlex.split(c) if c else ['bash', paths.root(*BLENDER)]


def work_dir(*p):
    return os.path.join(os.environ.get('HAIRDECK_VRM_WORK') or paths.results(WORK), *p)


def start_worker(body=None):
    """Start blender/vrm_worker.py for the body (one Blender process that keeps running and takes export_vrm's Blender
    step; it stops when this process is gone or work_dir('jobs', 'stop') exists). Its first start for a body builds the
    cache in work_dir('ref') (~30 s). Returns the Popen."""
    jobs = work_dir('jobs')
    os.makedirs(jobs, exist_ok=True)
    for f in ('ready', 'stop'):
        if os.path.exists(f'{jobs}/{f}'):
            os.remove(f'{jobs}/{f}')
    lf = open(work_dir('worker.log'), 'a')
    return subprocess.Popen(blender_cmd() + ['-b', '--python', os.path.join(SCRIPTS, 'vrm_worker.py'), '--', B.body_path(body),
                                             jobs, work_dir('ref'), str(os.getpid())],
                            stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=_env(False), cwd=RELEASE,
                            start_new_session=True)


def worker_pid(body=None):
    """PID of a running worker for the body (default body), else None."""
    try:
        r = json.load(open(work_dir('jobs', 'ready')))
        os.kill(r['pid'], 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return r['pid'] if r.get('body') == B.body_path(body) else None


def _blender(prep, d, body, timeout=300):
    """The Blender step through the worker when one is ready for the body, else one Blender run of
    blender/vrm_export.py. Returns (seconds, 'worker' | 'one-shot')."""
    t = time.time()
    if worker_pid(body):
        jobs = work_dir('jobs')
        jid = f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}'
        json.dump(dict(prep=prep, out=d), open(f'{jobs}/{jid}.tmp', 'w'))
        os.replace(f'{jobs}/{jid}.tmp', f'{jobs}/{jid}.json')
        done = f'{jobs}/{jid}.done.json'
        while not os.path.exists(done):
            if time.time() - t > timeout or not worker_pid(body):
                raise RuntimeError('VRM worker did not answer; see ' + work_dir('worker.log'))
            time.sleep(0.2)
        r = json.load(open(done))
        os.remove(done)
        if not r['ok']:
            raise RuntimeError('VRM worker: ' + r['error'])
        return time.time() - t, 'worker'
    cmd = blender_cmd() + ['-b', '--python', os.path.join(SCRIPTS, 'vrm_export.py'), '--', B.body_path(body), prep, d]
    return _run(cmd, f'{d}/blender.log', _env(False)), 'one-shot'


def _env(gl=True):
    """Environment of the subprocesses: this one without PYTHONPATH (HAIRDECK_PYTHONPATH, if set, takes its place), two
    OpenMP / MKL threads, no user site-packages; for the renders (gl) the OpenGL platform of HAIRDECK_GL."""
    e = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH',)}
    if os.environ.get('HAIRDECK_PYTHONPATH'):
        e['PYTHONPATH'] = os.environ['HAIRDECK_PYTHONPATH']
    e.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONNOUSERSITE='1')
    if gl and os.environ.get('HAIRDECK_GL') == 'osmesa':  # CPU rendering (no GPU, e.g. outside a GPU slot)
        e['PYOPENGL_PLATFORM'] = 'osmesa'
        e.pop('EGL_DEVICE_ID', None)
    elif gl:
        e.update(PYOPENGL_PLATFORM='egl', EGL_DEVICE_ID=os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0] or '0')
    return e


def _run(cmd, log, env, timeout=600):
    """Run one subprocess (working directory: the release tree) with its output in log; raises with the log's tail."""
    t = time.time()
    with open(log, 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=RELEASE)
    if r.returncode != 0:
        tail = open(log, errors='replace').read()[-1500:]
        name = cmd[cmd.index('--python') + 1] if '--python' in cmd else cmd[cmd.index('-m') + 1] if '-m' in cmd else cmd[0]
        raise RuntimeError(f'{os.path.basename(name)} failed (exit {r.returncode}): {tail}')
    return time.time() - t


def _last_json(log, tag):
    for line in reversed(open(log, errors='replace').read().splitlines()):
        if line.startswith(tag + ' '):
            return json.loads(line[len(tag) + 1:])
    return None


def patch_render_queue(path, ref):
    """The worker exports from a saved .blend, and the VRM add-on does not keep a material's imported VRM 0.x
    renderQueue through a .blend save (an eyelash material: 4500 in the source and in a one-shot export, 3000 after the
    .blend). Copy renderQueue from the add-on's re-export of the source (same session as the import) for every material
    of the same name; the JSON chunk is rewritten, the binary chunk is untouched. Returns the patched (name, old, new)."""
    b = open(path, 'rb').read()
    jl = struct.unpack('<I', b[12:16])[0]
    J = json.loads(b[20:20 + jl])
    rest = b[20 + jl:]
    R = json.loads(open(ref, 'rb').read()[20:20 + struct.unpack('<I', open(ref, 'rb').read()[12:16])[0]])
    want = {m.get('name'): m.get('renderQueue') for m in R['extensions']['VRM']['materialProperties']}
    changed = []
    for m in J['extensions']['VRM']['materialProperties']:
        nm = m.get('name')
        if nm in want and want[nm] is not None and m.get('renderQueue') != want[nm]:
            changed.append((nm, m.get('renderQueue'), want[nm]))
            m['renderQueue'] = want[nm]
    if changed:
        js = json.dumps(J, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        js += b' ' * (-len(js) % 4)
        body = struct.pack('<II', len(js), 0x4E4F534A) + js + rest
        open(path + '.tmp', 'wb').write(struct.pack('<III', 0x46546C67, 2, 12 + len(body)) + body)
        os.replace(path + '.tmp', path)
    return changed


def vrm_path(out_dir, seed, body=None):
    """The downloadable VRM of one sample (<file_tag>_seed<s>.vrm)."""
    return f"{out_dir}/{B.body(body)['file_tag']}_seed{seed}.vrm"


def export_vrm(out_dir, seed, colour_srgb, body=None, render=True):
    """One saved sample -> its VRM (see the module docstring). colour_srgb: hair colour, sRGB 0..1 (the image pipeline's
    info.json 'hair_color'); body: a body file (default hairdeck.vrm.body.default_path()). Returns dict(vrm, rest_png,
    ok, problems (of hairdeck.vrm.check), n_cards, n_chains_in, n_chains, n_spring_groups, hair_verts, removed_hair,
    bones_deleted, blender_sec, blender_path ('worker' | 'one-shot'), render_queue_patched, size_mb, timings)."""
    out_dir = os.path.abspath(out_dir)  # the subprocesses run in the release tree
    bd = B.body(body)
    T = {}
    t_all = time.time()
    d = f'{out_dir}/vrm_seed{seed}'
    run, prep = f'{d}/run', f'{d}/prep'
    os.makedirs(prep, exist_ok=True)
    t = time.time()
    n_cards, n_chains = prepare(out_dir, seed, bd, srgb_to_linear(colour_srgb), run, prep, f'{d}/prep.log')
    T['prep'] = time.time() - t
    gen = f"{d}/{B.file_name('gen', bd)}"
    if os.path.exists(gen):  # the VRM of an earlier export of this sample
        os.remove(gen)
    T['blender'], how = _blender(prep, d, body)
    if not os.path.exists(gen):
        raise RuntimeError('Blender finished without writing the VRM; see ' + f'{d}/blender.log')
    ref, orig = work_dir('ref', B.file_name('orig', bd)), f"{d}/{B.file_name('orig', bd)}"
    patched = patch_render_queue(gen, ref) if how == 'worker' and os.path.exists(ref) else []
    if os.path.exists(ref) and not os.path.exists(orig):
        os.symlink(ref, orig)
    t = time.time()
    chk = check(d, prep, bd)
    json.dump(chk, open(f'{d}/check.json', 'w'), indent=1, default=str)
    T['check'] = time.time() - t
    if render:
        T['render'] = _run([sys.executable, '-m', 'hairdeck.vrm.render', f'{d}/rest.png', gen], f'{d}/render.log', _env(True))
    dst = vrm_path(out_dir, seed, body)
    if os.path.exists(dst):
        os.remove(dst)
    os.link(gen, dst)
    ex = json.load(open(f"{d}/{B.file_name('export', bd)}"))
    T['total'] = time.time() - t_all
    info = dict(vrm=dst, rest_png=f'{d}/rest.png' if render else None, ok=chk['ok'], problems=chk['problems'],
                n_cards=n_cards, n_chains_in=n_chains, n_chains=ex.get('n_new_chains'), n_spring_groups=ex.get('n_spring_groups'),
                hair_verts=ex.get('hair_verts'), removed_hair=ex.get('hair_objects'), bones_deleted=ex.get('bones_deleted'),
                blender_sec=ex.get('blender_sec'), blender_path=how, render_queue_patched=patched,
                size_mb=round(os.path.getsize(dst) / 1e6, 1), timings={k2: round(v, 2) for k2, v in T.items()})
    json.dump(info, open(f'{d}/vrm_info.json', 'w'), indent=1, default=str)
    return info


def inputs_sha256(files, *extra):
    """sha256 over the bytes of the files (a missing one counts as absent) and the extra strings: what an output that is
    kept between calls was made from."""
    h = hashlib.sha256()
    for f in files:
        h.update(open(f, 'rb').read() if os.path.exists(f) else b'-')
        h.update(b'\0')
    for x in extra:
        h.update(str(x).encode() + b'\0')
    return h.hexdigest()


def turn_preview(out_dir, seed, colour_srgb, show_chains=False, body=None):
    """out_dir/turn_seed{s}.mp4 (turn_seed{s}_chains.mp4 with the chain joints drawn) from cards_seed{s}.json and
    chains_seed{s}.json (none if missing); an existing video is reused (cached=True) when its json records the same
    inputs (inputs_sha256 of the cards, the chains, the body file and the colour). Returns the summary of
    hairdeck.vis.turn_preview plus wall_sec and inputs_sha256."""
    t = time.time()
    out_dir = os.path.abspath(out_dir)
    tag = f'turn_seed{seed}' + ('_chains' if show_chains else '')
    out = f'{out_dir}/{tag}.mp4'
    cards, ch, body_file = f'{out_dir}/cards_seed{seed}.json', f'{out_dir}/chains_seed{seed}.json', B.body_path(body)
    colour = ','.join('%.6f' % x for x in colour_srgb)
    key = inputs_sha256([cards, ch, body_file], colour)
    if os.path.exists(out) and os.path.exists(f'{out_dir}/{tag}.json'):
        prev = json.load(open(f'{out_dir}/{tag}.json'))
        if prev.get('inputs_sha256') == key:
            return dict(prev, cached=True)
    cmd = [sys.executable, '-m', 'hairdeck.vis.turn_preview', cards, ch if os.path.exists(ch) else '-', colour, out,
           '--body', body_file] + (['--chains_visible'] if show_chains else [])
    _run(cmd, f'{out_dir}/{tag}.log', _env(True))
    res = _last_json(f'{out_dir}/{tag}.log', 'PREVIEW') or {}
    res['wall_sec'] = round(time.time() - t, 2)
    res['inputs_sha256'] = key
    json.dump(res, open(f'{out_dir}/{tag}.json', 'w'), indent=1)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description='VRM export and head-turn preview of a saved image-pipeline result')
    ap.add_argument('out_dir')
    ap.add_argument('seed', type=int)
    ap.add_argument('what', nargs='?', default='both', choices=['vrm', 'turn', 'both'])
    ap.add_argument('--body', default=None, help='body file (default: hairdeck.vrm.body.default_path())')
    a = ap.parse_args()
    info = json.load(open(f'{a.out_dir}/info.json'))
    smp = [s for s in info.get('samples', []) if s['seed'] == a.seed]
    if not smp or not smp[0].get('n_meshed'):
        sys.exit(f'{a.out_dir}: its info.json has no generated hair for seed {a.seed} ({info.get("message") or "not sampled"})')
    col = info['hair_color']
    if a.what in ('vrm', 'both'):
        print('VRM', json.dumps(export_vrm(a.out_dir, a.seed, col, a.body), default=str))
    if a.what in ('turn', 'both'):
        print('TURN', json.dumps(turn_preview(a.out_dir, a.seed, col, body=a.body)))


if __name__ == '__main__':
    main()
