"""The training data of all models from the CHARM dataset and the characters' VRM files, in six steps run in order;
every step skips the work whose outputs exist, so a stopped run continues where it stopped:
  cards         the index of the dataset (hairdeck.data.index), the card templates extracted from the original hair
                meshes without CHARM code (hairdeck.cards.extract; bitwise the data the released models were trained
                on), the fixed-K card features of the set DiTs (hairdeck.cards.codec.hair_features), the card token
                stats (hairdeck.models.tokenizer.compute_stats) and the test lists of data/lists for the evaluation
  render_lists  every indexed character with a VRM file
  renders       the image-condition renders (blender/gen_data.py) in parallel Blender processes
  dora          the frozen Dora-VAE 1.1 encoder tokens of the point-cloud DiT condition (hairdeck.encoders.dora)
  head          the head frame of the image DiT and the chain DiT (hairdeck.data.head_frame): head fits, k and b of
                p_head = k * p + b per hairstyle, the token stats in the head frame
  chains        the spring-bone chains of the VRM files (hairdeck.spring.vrm_spring) and their tokens in the head frame
                (hairdeck.spring.chain_data)
Then scripts/train/launch_vae.sh trains the card VAE and writes the card latents of both frames.

python data.py DATASET_DIR VRM_DIR [VRM_DIR ...] [--steps cards,render_lists,renders,dora,head,chains] [--workers 12]
       [--render_workers 8] [--device cuda]
DATASET_DIR holds vroid-hair, vroid-hair-new and vroid-hair-tempconb (the id lists, hair.json, hair.glb and
hair_reform.ply of every hairstyle); a VRM_DIR holds the VRM file of the character [prefix/]shard/id at
<VRM_DIR>/<shard>/<id>/<id>.vrm, and a character found in several VRM_DIRs takes the first. Both are read-only.
Needs the Dora-VAE 1.1 weights and Blender 4.2 LTS with the VRM Add-on for Blender (scripts/setup/download_weights.py
--only dora blender). Writes into the data root (hairdeck.paths), per split (train, test):
  cards         {split}_meta.json        the index (row i = row i of the condition caches), index_report.txt
                {split}_templates.npz    the cards of every indexed hairstyle with a hair.glb: points (P,5) float32,
                                         card_ptr, card_shape uint8, card_nrm (C,3) float32, hair_ptr;
                                         {split}_cards_meta.json (the hairstyles extracted, in order),
                                         {split}_extract_fails.json
                cards_{split}.npz        res (C,K,5) f32, the K = 64 point arc-length resample of every card; nrm (C,3)
                                         f32, its root width direction pointing away from the hairstyle's mean root;
                                         n (C,) i32; shape (C,) u8; hair_ptr (H+1,) i64; cond_idx (H,), the row of
                                         every hairstyle in {split}_meta.json
                token_stats.json         root, width, thickness and point-count stats of the train cards
                test_pc_orig_idx.json, test_pc_orig_idx_all.json, test_idxs_rendered.json, test_keys.txt
  render_lists  render_list_{split}.txt  'key<TAB>VRM file' of every character of {split}_meta.json with a VRM file
                                         (the others are left out and counted)
  renders       renders/ (train), renders_test/ (test): per character <key with '/' -> '_'>/v<i>.png (RGBA),
                v<i>_lab.png (labels), meta.json of 8 views (seed 0, 768 px); _failed/<key>.txt per failed character,
                _shards/w<i>.txt and _logs/w<i>.log per Blender process
  dora          cond_dora_{split}.npy    float16 [N, 512, 768], row i = {split}_meta.json[i]: the tokens of its
                                         hair_reform.ply (10,000 points, xyz * 2 and unit normals, no per-sample
                                         normalisation; the test files are byte-equal to CHARM's released test point
                                         clouds); both encoder branches get the same points, queries = FPS from index
                                         0 of 256 points per branch, batches of 16, fp32 forward with TF32 off;
                                         cond_dora_info.json
  head          head_params_{split}.json the head fit of every rendered character with a hair.glb (face bbox, head
                                         joint and the Blender -> dataset similarity, from the render meta.json and the
                                         hair mesh); prints the fit diagnostics
                head_frame_{split}.npz   valid, k, b per hairstyle (train_cards_meta / test_meta order) and the
                                         canonical head W, O: the medians of the train fits, or W, O of the other
                                         head_frame npz if one exists
                token_stats_head.json    token_stats.json with root mu / sd recomputed in the head frame
  chains        spring/chains_{split}.jsonl  the hair spring-bone chains of every character with a VRM file and a
                                         hair.glb in the dataset frame, one JSON line per character: key, status, fit
                                         (similarity glTF -> dataset, residual), n_groups, n_chains_all, n_hair_verts,
                                         frac_hair_verts_on_chains, chains (positions, node names, stiffness / gravity
                                         / drag / hit, n_hair_verts); characters already in the file are skipped
                spring/chains_{split}_head.npz, spring/chain_stats_head.json  the chain tokens in the head frame and
                                         their stats over the train chains
--steps: the steps to run (comma-separated). --workers: CPU processes. --render_workers: Blender processes (4 threads
each, Blender restarted every 40 characters; a character in progress when Blender died or the run was stopped counts
as failed). HAIRDECK_BLENDER: the command that runs Blender, split like a shell command line (default bash
ROOT/third_party/blender/bl.sh, as for hairdeck.vrm.export). --device: the dora step (one GPU).
W, O, the token stats and the chain stats of the released models come with the checkpoints (data/head_frame_train.npz,
data/token_stats.json, data/token_stats_head.json, data/spring/chain_stats_head.json); kept in place, they make the
data those of the released models (and scripts/train/export_latents.py card_vae --keep_stats the card latents).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

RELEASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RELEASE)
from hairdeck import paths  # noqa: E402
from hairdeck.cards.codec import hair_features  # noqa: E402
from hairdeck.cards.extract import pack, work  # noqa: E402
from hairdeck.cards.packed import Packed  # noqa: E402
from hairdeck.data import index  # noqa: E402
from hairdeck.data.head_frame import canonical_head, card_hair, fit_one, items_of, params, root_stats  # noqa: E402
from hairdeck.spring.chain_data import M, build, stats_of  # noqa: E402
from hairdeck.spring.vrm_spring import extract_record  # noqa: E402
from hairdeck.vrm.export import blender_cmd  # noqa: E402

SPLITS = ('train', 'test')
STEPS = ('cards', 'render_lists', 'renders', 'dora', 'head', 'chains')
LISTS = ('test_pc_orig_idx.json', 'test_pc_orig_idx_all.json', 'test_idxs_rendered.json', 'test_keys.txt')
K = 64  # arc-length samples per card of the set DiT data
HAIRS = {'train': 'train_cards_meta.json', 'test': 'test_meta.json'}  # the hairstyle orders of the head frame and chain data
RENDERS = {'train': 'renders', 'test': 'renders_test'}
VIEWS, BLENDER_THREADS, CHARS_PER_PROCESS = 8, 4, 40  # the released renders; Blender restarts for memory hygiene
BLENDER_LOG = re.compile(r'^(CHAR|FAIL|WORKER_EXIT)|Traceback|Error')
G = {}


def have(*fns):
    return all(os.path.isfile(paths.data(f)) for f in fns)


def load(fn):
    return json.load(open(paths.data(fn)))


def replace_with(fn, write, mode='w'):
    """Write fn through a temporary file, so a stopped run leaves no partial output."""
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn + '.partial', mode) as f:
        write(f)
    os.replace(fn + '.partial', fn)


def save_json(fn, obj):
    replace_with(paths.data(fn), lambda f: json.dump(obj, f))


def save_npz(fn, **arrays):
    replace_with(paths.data(fn), lambda f: np.savez(f, **arrays), 'wb')


# ----------------------------------------------------------------------------- cards
def features(i):
    res, nrm = hair_features(G['P'].cards(i), K)
    return i, res, nrm


def extract_split(split, workers):
    meta = load(f'{split}_meta.json')
    items = [(m['key'], m['dir']) for m in meta if os.path.isfile(os.path.join(m['dir'], 'hair.glb'))]
    os.environ['OMP_NUM_THREADS'] = '1'
    res, fails = {}, []
    with Pool(workers) as pool:
        for n, (key, d, cards, err) in enumerate(pool.imap(work, items, chunksize=4)):
            if cards is None:
                fails.append((key, err))
            else:
                res[key] = (d, cards)
            if (n + 1) % 1000 == 0:
                print(n + 1, 'done', len(fails), 'failed', flush=True)
    done = [k for k, _ in items if k in res]
    arrays, mt, status = pack(done, res)
    save_npz(f'{split}_templates.npz', **arrays)
    save_json(f'{split}_cards_meta.json', mt)
    save_json(f'{split}_extract_fails.json', fails)
    print(f'{split}: {len(done)} hairstyles, {len(arrays["card_shape"])} cards, {len(fails)} failed hairstyles; card status {status}', flush=True)


def features_split(split, workers):
    P = Packed(paths.data(f'{split}_templates.npz'))
    G['P'] = P
    C = len(P.card_ptr) - 1
    res = np.zeros((C, K, 5), np.float32)
    nrm = np.zeros((C, 3), np.float32)
    with Pool(workers) as pool:
        for i, r, v in pool.imap_unordered(features, range(len(P)), chunksize=32):
            a, b = P.hair_ptr[i], P.hair_ptr[i + 1]
            res[a:b], nrm[a:b] = r, v
    n = np.diff(P.card_ptr).astype(np.int32)
    pos = {m['key']: i for i, m in enumerate(load(f'{split}_meta.json'))}  # the order of the condition caches
    cond_idx = np.array([pos[m['key']] for m in load(f'{split}_cards_meta.json')], np.int64)
    save_npz(f'cards_{split}.npz', res=res, nrm=nrm, n=n, shape=P.card_shape, hair_ptr=P.hair_ptr, cond_idx=cond_idx)
    print(split, 'cards', C, 'hairs', len(P), flush=True)


def step_cards(dataset_dir, workers):
    if have(*[f'{s}_meta.json' for s in SPLITS], 'index_report.txt'):
        print('index: kept', flush=True)
    else:
        metas, report = index.build(dataset_dir, SPLITS, workers)
        for split, meta in metas.items():
            save_json(f'{split}_meta.json', meta)
        txt = '\n'.join(report)
        print(txt, flush=True)
        replace_with(paths.data('index_report.txt'), lambda f: f.write(txt + '\n'))
    for f in LISTS:
        if not have(f):
            shutil.copyfile(os.path.join(RELEASE, 'data', 'lists', f), paths.data(f))
    for split in SPLITS:
        if have(f'{split}_templates.npz', f'{split}_cards_meta.json', f'{split}_extract_fails.json'):
            print(f'{split} templates: kept', flush=True)
        else:
            extract_split(split, workers)
        if have(f'cards_{split}.npz'):
            print(f'{split} card features: kept', flush=True)
        else:
            features_split(split, workers)
    if not have('token_stats.json'):  # what the point-cloud DiT's tokenizer would write; the head-frame stats start from it
        from hairdeck.models.tokenizer import compute_stats  # torch
        z = np.load(paths.data('cards_train.npz'))
        st = compute_stats(dict(res=z['res'], n=z['n']))
        save_json('token_stats.json', {k: np.asarray(v).tolist() for k, v in st.items()})


# ----------------------------------------------------------------------------- render lists
def vrm_file(key, vrm_dirs):
    shard, i = key.split('/')[-2:]
    for d in vrm_dirs:
        fn = os.path.join(d, shard, i, f'{i}.vrm')
        if os.path.isfile(fn):
            return fn
    return None


def step_render_lists(vrm_dirs):
    dirs = [os.path.abspath(d) for d in vrm_dirs]
    for split in SPLITS:
        fn = f'render_list_{split}.txt'
        if have(fn):
            print(f'{fn}: kept', flush=True)
            continue
        meta = load(f'{split}_meta.json')
        found = [(m['key'], vrm_file(m['key'], dirs)) for m in meta]
        lines = [f'{k}\t{v}\n' for k, v in found if v]
        per_dir = ', '.join(f'{sum(v is not None and v.startswith(d + os.sep) for _, v in found)} in {d}' for d in dirs)
        replace_with(paths.data(fn), lambda f: f.write(''.join(lines)))
        print(f'{fn}: {len(lines)} of {len(meta)} characters ({per_dir}); {len(meta) - len(lines)} without a VRM file, left out', flush=True)


# ----------------------------------------------------------------------------- renders
def blender_shard(cmd, log_fn, env, count):
    """Blender on one shard until gen_data.py finds nothing left to render (WORKER_EXIT 0). A Blender process that dies
    without WORKER_EXIT is restarted after 5 s (gen_data.py then marks the character it died on as failed); after three
    such deaths in a row without a CHAR or FAIL line (Blender does not start) the shard stops and False is returned."""
    silent = 0
    with open(log_fn, 'a') as log:
        while True:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                 text=True, errors='replace')
            tail, exit_line, chars = collections.deque(maxlen=20), None, 0
            for line in p.stdout:
                tail.append(line)
                if BLENDER_LOG.search(line):
                    log.write(line)
                    log.flush()
                    if line.startswith('WORKER_EXIT'):
                        exit_line = line.strip()
                    elif line.startswith(('CHAR', 'FAIL')):
                        chars += 1
                        count(line.startswith('FAIL'))
            p.wait()
            if exit_line == 'WORKER_EXIT 0':
                break
            silent = 0 if exit_line or chars else silent + 1
            if exit_line is None:
                log.write(f'BLENDER_DIED {time.ctime()}\n')
                if silent == 3:
                    log.write('last output:\n' + ''.join(tail))
                    return False
                time.sleep(5)
        log.write(f'SHARD_DONE {time.ctime()}\n')
    return True


def render_split(split, workers):
    out = paths.data(RENDERS[split])
    items = [l.rstrip('\n').split('\t')[:2] for l in open(paths.data(f'render_list_{split}.txt')) if l.strip()]

    def done(key):
        k_ = key.replace('/', '_')
        return os.path.exists(os.path.join(out, k_, 'meta.json')) or os.path.exists(os.path.join(out, '_failed', f'{k_}.txt'))
    todo = [it for it in items if not done(it[0])]
    if not todo:
        n_failed = len(os.listdir(os.path.join(out, '_failed'))) if os.path.isdir(os.path.join(out, '_failed')) else 0
        print(f'renders {split}: all {len(items)} characters done ({n_failed} failed)', flush=True)
        return
    n = min(workers, len(todo))
    print(f'renders {split}: {len(todo)} of {len(items)} characters in {n} Blender processes -> {out}', flush=True)
    for sub in ('_shards', '_logs'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    lock, seen = threading.Lock(), [0, 0]

    def count(failed):
        with lock:
            seen[0] += 1
            seen[1] += failed
            if seen[0] % 100 == 0 or seen[0] == len(todo):
                print(f'renders {split}: {seen[0]} / {len(todo)} ({seen[1]} failed)', flush=True)

    def shard(w):
        fn = os.path.join(out, '_shards', f'w{w}.txt')
        with open(fn, 'w') as f:
            f.writelines(f'{k}\t{v}\n' for k, v in todo[w::n])
        cmd = blender_cmd() + ['-b', '-t', str(BLENDER_THREADS), '--python', os.path.join(RELEASE, 'blender', 'gen_data.py'),
                               '--', out, fn, '--views', str(VIEWS), '--max_chars', str(CHARS_PER_PROCESS)]
        return blender_shard(cmd, os.path.join(out, '_logs', f'w{w}.log'), env, count)
    with ThreadPoolExecutor(n) as ex:
        ok = list(ex.map(shard, range(n)))
    if not all(ok):
        raise RuntimeError(f'renders {split}: Blender did not start in shards {[w for w in range(n) if not ok[w]]} '
                           f'(see {out}/_logs; HAIRDECK_BLENDER = the command that runs Blender)')
    print(f'renders {split}: done ({seen[1]} failed)', flush=True)


def step_renders(workers):
    for split in SPLITS:
        render_split(split, workers)


# ----------------------------------------------------------------------------- dora
def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def step_dora(workers, device):
    from hairdeck.encoders.common import meta_dirs  # torch
    from hairdeck.encoders.dora import BATCH, COND_SHAPE, DORA_WEIGHTS, condition_tokens
    for split in SPLITS:
        out_fn = paths.data(f'cond_dora_{split}.npy')
        if os.path.isfile(out_fn):
            print(f'cond_dora_{split}.npy: kept', flush=True)
            continue
        plys = [os.path.join(d, 'hair_reform.ply') for d in meta_dirs(split)]
        n = len(plys)
        out = np.lib.format.open_memmap(out_fn + '.partial', mode='w+', dtype=np.float16, shape=(n,) + COND_SHAPE)
        t0 = time.time()
        condition_tokens(split, out, device, workers, log=lambda s: print(split, s, f'{time.time() - t0:.0f}s', flush=True), plys=plys)
        out.flush()
        del out
        dt = time.time() - t0
        os.replace(out_fn + '.partial', out_fn)
        print(f'cond_dora_{split}.npy: {n} rows {dt:.0f}s ({1000 * dt / n:.1f} ms/sample)', flush=True)

        info_fn = paths.data('cond_dora_info.json')
        info = json.load(open(info_fn)) if os.path.isfile(info_fn) else {}
        T, C = COND_SHAPE
        info.update(encoder='Dora-VAE 1.1 encoder (Seed3D/Dora), forward pass re-implemented in hairdeck/encoders/dora.py',
                    hf='Seed3D/Dora-VAE-1.1 @ 7187b01c78e1bd4a2ddf8301f145f5dc98ac2c22', licence_weights='Apache-2.0',
                    sharp_branch_input='copy', stored='pre-KL tokens: encoder output after encoder.ln_post (width 768)',
                    weights_sha256=sha256(paths.weights(*DORA_WEIGHTS)), T=T, C=C, dtype='float16',
                    precision='fp32 forward, TF32 off, stored as float16')
        info.setdefault('splits', {})[split] = dict(shape=[n, T, C], rows='data/%s_meta.json order' % split,
                                                    seconds=round(dt, 1), ms_per_sample=round(1000 * dt / n, 2), batch_size=BATCH)
        replace_with(info_fn, lambda f: json.dump(info, f, indent=1))


# ----------------------------------------------------------------------------- head frame
def fit_report(res):
    print({s: sum(r['status'] == s for r in res.values()) for s in ('ok', 'no_face', 'error')})
    ok = [r for r in res.values() if r['status'] == 'ok']
    if not ok:
        return
    sa = np.array([r['s_axes'] for r in ok])
    print('per-axis scale spread (max/min): median %.4f, p95 %.4f, p99 %.4f' % tuple(np.percentile(sa.max(1) / sa.min(1), [50, 95, 99])))
    print('bbox residual / hair size: median %.4f, p95 %.4f' % tuple(np.percentile([r['resid'] for r in ok], [50, 95])))
    print('face width (dataset units): p5 %.4f median %.4f p95 %.4f' % tuple(np.percentile([r['w'] for r in ok], [5, 50, 95])))
    o = np.array([r['o'] for r in ok])
    print('face centre: median', np.median(o, 0).round(4), 'std', o.std(0).round(4), flush=True)


def fit_split(split, workers):
    items = items_of(load(HAIRS[split]), paths.data(RENDERS[split]))
    print(split, len(items), 'rendered characters with hair.glb', flush=True)
    if workers > 1:
        with Pool(workers) as p:
            res = dict(p.map(fit_one, items, chunksize=16))
    else:
        res = dict(map(fit_one, items))
    save_json(f'head_params_{split}.json', res)
    fit_report(res)
    return res


def step_head(workers):
    fits = {}
    for split in SPLITS:
        if have(f'head_params_{split}.json'):
            print(f'head_params_{split}.json: kept', flush=True)
            fits[split] = load(f'head_params_{split}.json')
        else:
            fits[split] = fit_split(split, workers)
    there = [s for s in SPLITS if have(f'head_frame_{s}.npz')]
    if len(there) < len(SPLITS):
        if there:  # one head frame in place (e.g. the released one): its canonical head
            z = np.load(paths.data(f'head_frame_{there[0]}.npz'))
            W, O = float(z['W']), z['O']
        else:
            W, O = canonical_head(fits['train'])
        n_ok = sum(f['status'] == 'ok' for f in fits['train'].values())
        print('canonical head: width %.4f centre %s (%s)' % (W, O.round(4), f'of head_frame_{there[0]}.npz' if there else f'{n_ok} train fits'))
        for split in SPLITS:
            if split in there:
                print(f'head_frame_{split}.npz: kept', flush=True)
                continue
            valid, k, b = params(load(HAIRS[split]), fits[split], W, O)
            save_npz(f'head_frame_{split}.npz', valid=valid, k=k, b=b, W=W, O=O)
            print('%s valid %d / %d, k p5/p50/p95 %s' % (split, valid.sum(), len(valid), np.percentile(k[valid], [5, 50, 95]).round(3)), flush=True)
    else:
        print('head_frame_{train,test}.npz: kept', flush=True)
    if have('token_stats_head.json'):
        print('token_stats_head.json: kept', flush=True)
        return
    hz, c = np.load(paths.data('head_frame_train.npz')), np.load(paths.data('cards_train.npz'))
    card_h = card_hair(c['hair_ptr'])
    kc, vc = hz['k'][card_h], hz['valid'][card_h]
    roots = c['res'][:, 0, :3]
    roots_h = kc[:, None] * roots + hz['b'][card_h]
    save_json('token_stats_head.json', root_stats(load('token_stats.json'), roots_h, vc))
    print('root sd dataset frame %s -> head frame %s' % (roots[vc].std(0).round(4), roots_h[vc].std(0).round(4)), flush=True)


# ----------------------------------------------------------------------------- chains
def chains_split(split, workers):
    key2dir = {e['key']: e['dir'] for e in load(HAIRS[split])}
    out = paths.data('spring', f'chains_{split}.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done, good, cut = set(), [], 0
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)['key'])
                good.append(line)
            except (json.JSONDecodeError, KeyError):  # a line cut short by a stopped run
                cut += line.strip() != ''
        if cut:
            replace_with(out, lambda f: f.writelines(good))
            print(f'chains {split}: {cut} incomplete line(s) of a stopped run dropped', flush=True)
    jobs = []
    for line in open(paths.data(f'render_list_{split}.txt')):
        if not line.strip():
            continue
        key, vrm = line.rstrip('\n').split('\t')
        if key in done or key not in key2dir:
            continue
        jobs.append((key, vrm, f'{key2dir[key]}/hair.glb'))
    print(f'chains {split}: {len(jobs)} characters to extract, {len(done)} done', flush=True)
    if not jobs:
        return
    n_ok = 0
    with Pool(workers) as p, open(out, 'a') as f:
        for k, r in enumerate(p.imap_unordered(extract_record, jobs, chunksize=4)):
            f.write(json.dumps(r) + '\n')
            n_ok += r['status'] == 'ok'
            if (k + 1) % 500 == 0:
                f.flush()
                print(f'{k + 1}/{len(jobs)} ok {n_ok}', flush=True)
    print('extracted', split, 'ok', n_ok, 'of', len(jobs), flush=True)


def step_chains(workers):
    for split in SPLITS:
        chains_split(split, workers)
    built = {}
    for split in SPLITS:
        if have(f'spring/chains_{split}_head.npz'):
            print(f'chains_{split}_head.npz: kept', flush=True)
        else:
            built[split] = build(split)
    if have('spring/chain_stats_head.json'):
        print('chain_stats_head.json: kept', flush=True)
        return
    if 'train' in built:
        tok, cnt, valid = built['train']
    else:
        z = np.load(paths.data('spring', 'chains_train_head.npz'))
        tok, cnt, valid = z['tok'], z['count'], z['valid']
    stats, X = stats_of(tok, cnt, valid)
    save_json('spring/chain_stats_head.json', stats)
    print('train chains', len(X), '| joints mean %.2f | stiffness mean %.2f' % (X[:, M * 3].mean(), X[:, M * 3 + 1].mean()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('vrm_dirs', nargs='+', metavar='vrm_dir')
    ap.add_argument('--steps', default=','.join(STEPS), help='comma-separated, of ' + ','.join(STEPS))
    ap.add_argument('--workers', type=int, default=12, help='CPU processes')
    ap.add_argument('--render_workers', type=int, default=8, help='Blender processes')
    ap.add_argument('--device', default='cuda', help='the dora step')
    a = ap.parse_args()
    steps = [s for s in a.steps.split(',') if s]
    if set(steps) - set(STEPS):
        ap.error(f'unknown steps {sorted(set(steps) - set(STEPS))}; the steps: {",".join(STEPS)}')
    os.makedirs(paths.data(), exist_ok=True)
    run = dict(cards=lambda: step_cards(a.dataset_dir, a.workers), render_lists=lambda: step_render_lists(a.vrm_dirs),
               renders=lambda: step_renders(a.render_workers), dora=lambda: step_dora(a.workers, a.device),
               head=lambda: step_head(a.workers), chains=lambda: step_chains(a.workers))
    for s in STEPS:
        if s in steps:
            t0 = time.time()
            print(f'== {s}', flush=True)
            run[s]()
            print(f'== {s} done ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
