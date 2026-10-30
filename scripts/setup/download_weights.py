"""Download everything inference and the demo need (the default items) and, with --only, the files of the test-set
evaluation into the workspace root (hairdeck.paths), each file checked against its sha256 where the source is fixed:

  hairdeck    our checkpoints and data files (image model, card VAE, chain model, point-cloud model as safetensors,
              token statistics, the canonical head, the avatar body) from the Hugging Face model repository --repo:
              config.json first, then every file of its MANIFEST.json against its sha256, but those of the test_data
              entry of config.json
  birefnet    BiRefNet-matting (MIT), Hugging Face ZhengPeng7/BiRefNet-matting at a fixed revision: the weights and the
              repository's model code -> weights/birefnet_matting/
  sam3        SAM 3 (SAM License), Hugging Face facebook/sam3 sam3.pt -> weights/sam3/; a gated repository: accept the
              licence on its page and log in first (`hf auth login`)
  sam3_src    the SAM 3 code, facebookresearch/sam3 at a fixed commit -> third_party/sam3_src (or pip install it)
  dinov3_src  the DINOv3 code (torch.hub), facebookresearch/dinov3 at a fixed commit -> third_party/dinov3_src
  dinov3      DINOv3 ViT-L/16 LVD-1689M (DINOv3 License), Hugging Face facebook/dinov3-vitl16-pretrain-lvd1689m
              model.safetensors at a fixed revision -> weights/dinov3/; a gated repository like facebook/sam3
  blender     Blender 4.2.23 LTS (Linux x64, GPL) from download.blender.org -> third_party/blender/, the wrapper bl.sh,
              and the VRM Add-on for Blender 4.10.2 (GitHub release) installed into its own user extensions folder
Not in the default, for the test-set evaluation (scripts/test.py):
  test_data      the evaluation data of the test set from --repo (the files of the test_data entry of config.json,
                 against MANIFEST.json: the original cards and their width directions, GT chains, ...), and the test
                 index lists and keys of data/lists copied into data/ (as scripts/data.py does)
  charm_test_pc  CHARM's released test point clouds (pc/ of the Hugging Face dataset repository hyz317/CHARM at a fixed
                 revision, sha256 per file in configs/charm_test_pc.sha256) -> data/charm_test_pc/
  dora           Dora-VAE 1.1 (Apache-2.0), Hugging Face Seed3D/Dora-VAE-1.1 dora_vae_1_1.ckpt at a fixed revision,
                 converted once by scripts/setup/convert_dora_ckpt.py -> weights/dora_vae_1_1/
                 dora_vae_1_1_shape_model.safetensors (sha256 checked); the 2.2 GB checkpoint is removed afterwards
The DINOv3, SAM 3 and BiRefNet licences come with their downloads; read them before use.

python scripts/setup/download_weights.py [--root DIR] [--only NAME ...] [--repo REPO] [--from DIR]
Hugging Face downloads go to HF_ENDPOINT if it is set (as huggingface_hub does).
--from DIR takes the files from another workspace instead of the network (same layout, same checks; dora: the
converted file; blender: the unpacked build when DIR has no archive, unchecked like the source checkouts). Files go
where hairdeck.paths reads them (HAIRDECK_DATA, HAIRDECK_CKPT and HAIRDECK_WEIGHTS apply).
"""
import argparse
import hashlib
import http.client
import json
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hairdeck import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(HERE))
REPO = 'hyz317/HairDeck'  # the Hugging Face model repository of the release
BIREFNET = ('ZhengPeng7/BiRefNet-matting', 'eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9', {
    'model.safetensors': 'a9875de5b1e6c8eb5fdaa8c727a82927ce442cdc87ba3abee6a77e6fa46c25bb',
    'birefnet.py': '2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831',
    'BiRefNet_config.py': 'e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52',
    'config.json': 'b2b235983b80fbff325976ca517bb7eda25e915b84a97b150eca41266cf8a13b'})
SAM3 = ('facebook/sam3', 'sam3.pt', '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e')
SOURCES = {'sam3_src': ('facebookresearch/sam3', '0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc', 'sam3_src'),
           'dinov3_src': ('facebookresearch/dinov3', '6876159a11b4df116f30f667f8c9888617df0751', 'dinov3_src')}
DINOV3 = ('facebook/dinov3-vitl16-pretrain-lvd1689m', 'ea8dc2863c51be0a264bab82070e3e8836b02d51', 'model.safetensors',
          'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179')
BLENDER = ('https://download.blender.org/release/Blender4.2/blender-4.2.23-linux-x64.tar.xz',
           'bea0eb3146be13eae6225409a117b215184f41b7f79e799f97cb3abb8f6dc404', 'blender-4.2.23-linux-x64')
VRM_ADDON = ('https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/v4.10.2/'
             'VRM_Addon_for_Blender-Extension-4_10_2.zip',
             '228e622d158fbbf070b1b49547708f360a223873f00fa81a42170794492e6db9')
CHARM_PC = ('hyz317/CHARM', '2515ec969b6de4b32e1b6f011ede89b2a889ced9')  # dataset repository, revision
DORA_SHA256 = '8ff217e936c8ba3cf8164b7715785ca4d197b0425f1239ee0ad9f65b8ba7cd95'  # the converted safetensors file
TEST_LISTS = ('test_pc_orig_idx.json', 'test_pc_orig_idx_all.json', 'test_idxs_rendered.json', 'test_keys.txt')
DEFAULT = ('hairdeck', 'birefnet', 'sam3', 'sam3_src', 'dinov3_src', 'dinov3', 'blender')
ITEMS = DEFAULT + ('test_data', 'charm_test_pc', 'dora')


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def have(dst, want):
    return os.path.isfile(dst) and (want is None or sha256(dst) == want)


def check(dst, want):
    if want is not None:
        got = sha256(dst)
        if got != want:
            os.replace(dst, dst + '.bad')
            raise RuntimeError(f'sha256 mismatch for {dst}: {got} != {want} (kept as {dst}.bad)')


def ssl_context():
    """certifi's certificates when installed (a dependency of huggingface_hub; some Pythons, e.g. the python.org builds
    for macOS, have no system certificates), else the system's"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def hf_endpoint():
    """the Hugging Face endpoint, HF_ENDPOINT as huggingface_hub reads it (e.g. a mirror)"""
    return os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')


def url_get(url, dst, want=None, tries=3):
    if have(dst, want):
        log('present', dst)
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(tries):
        log('download', url, f'(try {i + 1} of {tries})' if i else '')
        try:
            with urllib.request.urlopen(url, timeout=60, context=ssl_context()) as r, open(dst + '.part', 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 22)
            break
        except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, dropped connections, IncompleteRead
            log('download failed:', f'{type(e).__name__}: {e}')
            if i == tries - 1:
                raise
            time.sleep(10)
    os.replace(dst + '.part', dst)
    check(dst, want)
    return dst


LAYOUT = (('data', 'data'), ('ckpt', 'ckpt'), ('weights', os.path.join('third_party', 'weights')))


def dst_of(rel):
    """The local file of rel, a path in the workspace layout (that of the model repository and of --from: data/...,
    ckpt/..., third_party/weights/..., the rest under the root), where hairdeck.paths reads it (HAIRDECK_DATA,
    HAIRDECK_CKPT and HAIRDECK_WEIGHTS apply)."""
    for key, base in LAYOUT:
        if rel.startswith(base + os.sep):
            return os.path.join(paths.get(key), rel[len(base) + 1:])
    return paths.root(rel)


def rel_of(dst):
    """The layout path of a local file (dst_of inverted): the file to take from a --from workspace."""
    for key, base in LAYOUT:
        d = paths.get(key).rstrip(os.sep)
        if dst.startswith(d + os.sep):
            return os.path.join(base, dst[len(d) + 1:])
    return os.path.relpath(dst, paths.get('root'))


def hub_get(repo, file, dst, want=None, revision=None, fresh=False):
    """file of a Hugging Face repository to dst (through local_dir, so no second copy in the Hugging Face cache; a dst
    that does not end with file, e.g. under HAIRDECK_DATA, gets the file moved there from the root)"""
    if not fresh and have(dst, want):
        log('present', dst)
        return dst
    from huggingface_hub import hf_hub_download
    log('hub', repo, file, revision or '')
    got = hf_hub_download(repo, file, revision=revision,
                          local_dir=dst[:-len(file)] if dst.endswith(os.sep + file) else paths.get('root'))
    if os.path.abspath(got) != os.path.abspath(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(got, dst)
    check(dst, want)
    return dst


def local_get(src_root, rel, dst, want=None, fresh=False):
    if not fresh and have(dst, want):
        log('present', dst)
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(os.path.join(src_root, rel), dst + '.part')
    os.replace(dst + '.part', dst)
    check(dst, want)
    log('copied', rel)
    return dst


def repo_index(root, a):
    """config.json and MANIFEST.json of the model repository (always the current ones) -> (config, MANIFEST files)"""
    out = []
    for f in ('config.json', 'MANIFEST.json'):
        dst = os.path.join(root, f)
        (local_get(a.src, f, dst, fresh=True) if a.src else hub_get(a.repo, f, dst, fresh=True))
        out.append(json.load(open(dst)))
    return out[0], out[1]['files']


def test_files(cfg, files):
    """The files of the test_data entry of config.json (an entry that ends with '/' is a folder)."""
    entries = cfg.get('data', {}).get('test_data', {}).get('files', [])
    return {rel for rel in files if any(rel == e or (e.endswith('/') and rel.startswith(e)) for e in entries)}


def get_files(root, a, rels, files):
    for rel in sorted(rels):
        dst = dst_of(rel)
        (local_get(a.src, rel, dst, files[rel]['sha256']) if a.src else hub_get(a.repo, rel, dst, files[rel]['sha256']))


def get_hairdeck(root, a):
    cfg, files = repo_index(root, a)
    get_files(root, a, set(files) - test_files(cfg, files), files)


def get_test_data(root, a):
    cfg, files = repo_index(root, a)
    rels = test_files(cfg, files)
    if not rels:
        raise RuntimeError('config.json of the model repository has no test_data entry')
    get_files(root, a, rels, files)
    os.makedirs(paths.data(), exist_ok=True)
    for f in TEST_LISTS:  # the evaluation reads them from data/
        shutil.copyfile(os.path.join(RELEASE, 'data', 'lists', f), paths.data(f))
    log('copied', ' '.join(TEST_LISTS), 'from data/lists')


def get_charm_test_pc(root, a):
    repo, rev = CHARM_PC
    want = {}
    for line in open(os.path.join(RELEASE, 'configs', 'charm_test_pc.sha256')):
        h, name = line.split()
        want[name] = h
    for name, h in sorted(want.items()):  # straight to data/charm_test_pc/<name> (not under pc/)
        dst = paths.data('charm_test_pc', name)
        if a.src:
            local_get(a.src, rel_of(dst), dst, h)
        else:
            url_get(f'{hf_endpoint()}/datasets/{repo}/resolve/{rev}/pc/{name}', dst, h)


def get_dora(root, a):
    from hairdeck.encoders.dora import DORA_CKPT, DORA_WEIGHTS
    repo, rev, f, want = DORA_CKPT
    dst = paths.weights(*DORA_WEIGHTS)
    if have(dst, DORA_SHA256):
        log('present', dst)
        return
    if a.src:
        local_get(a.src, rel_of(dst), dst, DORA_SHA256)
        return
    ck = hub_get(repo, f, paths.weights(DORA_WEIGHTS[0], f), want, revision=rev)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'convert_dora_ckpt.py'), '--src', ck, '--dst', dst])
    if r.returncode != 0:
        raise RuntimeError(f'convert_dora_ckpt.py failed ({ck} kept)')
    check(dst, DORA_SHA256)
    os.remove(ck)  # only the converted weights are read
    log('ready', dst, f'({f} removed after the conversion)')


def get_birefnet(root, a):
    repo, rev, files = BIREFNET
    for f, want in files.items():
        dst = paths.weights('birefnet_matting', f)
        (local_get(a.src, rel_of(dst), dst, want) if a.src else hub_get(repo, f, dst, want, revision=rev))


def get_sam3(root, a):
    repo, f, want = SAM3
    dst = paths.weights('sam3', f)
    (local_get(a.src, rel_of(dst), dst, want) if a.src else hub_get(repo, f, dst, want))
    if not a.src:
        try:
            hub_get(repo, 'LICENSE', paths.weights('sam3', 'LICENSE'))
        except Exception as e:  # noqa: BLE001
            log('SAM 3 LICENSE not fetched:', e)


def get_source(root, a, item):
    repo, commit, name = SOURCES[item]
    dst = paths.sam3_repo() if item == 'sam3_src' else paths.dinov3_repo()
    stamp = os.path.join(dst, '.hairdeck_commit')
    if os.path.isfile(stamp) and open(stamp).read().strip() == commit:
        log('present', dst)
        return
    if os.path.exists(dst) and not os.path.isfile(stamp):  # e.g. the user's own clone (HAIRDECK_*_REPO): left as it is
        log('present, not installed by this script, used as it is (the release was tested at', commit[:8] + '):', dst)
        return
    if a.src:
        shutil.copytree(os.path.join(a.src, 'third_party', name), dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.git', '__pycache__'))
    else:
        tgz = os.path.join(root, 'third_party', f'{name}-{commit[:8]}.tar.gz')
        url_get(f'https://github.com/{repo}/archive/{commit}.tar.gz', tgz)
        with tarfile.open(tgz) as t:
            top = t.getnames()[0].split('/')[0]
            t.extractall(os.path.dirname(dst), filter='data')
        if os.path.exists(dst):
            shutil.rmtree(dst)  # this script's checkout of an older commit (its stamp names another one)
        os.replace(os.path.join(os.path.dirname(dst), top), dst)
        os.remove(tgz)
    open(stamp, 'w').write(commit + '\n')
    log('ready', dst, commit[:8])


def get_dinov3(root, a):
    repo, rev, f, want = DINOV3
    dst = paths.weights('dinov3', f)
    (local_get(a.src, rel_of(dst), dst, want) if a.src else hub_get(repo, f, dst, want, revision=rev))
    if not a.src:
        try:
            hub_get(repo, 'LICENSE.md', paths.weights('dinov3', 'LICENSE.md'), revision=rev)
        except Exception as e:  # noqa: BLE001
            log('DINOv3 LICENSE.md not fetched:', e)


def get_blender(root, a):
    url, want, top = BLENDER
    B = os.path.join(root, 'third_party', 'blender')
    os.makedirs(B, exist_ok=True)
    if not os.path.isfile(os.path.join(B, top, 'blender')):
        tar = os.path.join(B, os.path.basename(url))
        src_build = os.path.join(a.src, 'third_party', 'blender', top) if a.src else ''
        if a.src and not os.path.isfile(os.path.join(a.src, os.path.relpath(tar, root))) and os.path.isdir(src_build):
            log('copy', src_build)  # an installed workspace keeps the unpacked build, not the archive
            shutil.copytree(src_build, os.path.join(B, top), symlinks=True)
        else:
            (local_get(a.src, os.path.relpath(tar, root), tar, want) if a.src else url_get(url, tar, want))
            log('extract', tar)
            with tarfile.open(tar) as t:
                t.extractall(B, filter='tar')
            os.remove(tar)
    shutil.copyfile(os.path.join(HERE, 'blender_wrapper.sh'), os.path.join(B, 'bl.sh'))
    addon = os.path.join(B, 'user', 'extensions', 'user_default', 'vrm')
    if os.path.isdir(addon):
        log('present', addon)
        return
    zurl, zwant = VRM_ADDON
    z = os.path.join(B, os.path.basename(zurl))
    if a.src:
        src = os.path.join(a.src, 'third_party', 'blender', os.path.basename(zurl))
        shutil.copyfile(src, z)
        check(z, zwant)
    else:
        url_get(zurl, z, zwant)
    r = subprocess.run(['bash', os.path.join(B, 'bl.sh'), '--command', 'extension', 'install-file', '--repo', 'user_default',
                        '--enable', z], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isdir(addon):
        raise RuntimeError('VRM Add-on install failed: ' + (r.stdout + r.stderr)[-2000:])
    log('VRM Add-on installed', addon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None, help='workspace root (default: hairdeck.paths root)')
    ap.add_argument('--only', nargs='*', default=list(DEFAULT), choices=ITEMS)
    ap.add_argument('--repo', default=REPO, help='Hugging Face model repository with our checkpoints')
    ap.add_argument('--from', dest='src', default='', help='copy from another workspace instead of downloading')
    a = ap.parse_args()
    if a.root:
        os.environ['HAIRDECK_ROOT'] = os.path.abspath(a.root)
    root = paths.get('root')
    os.makedirs(root, exist_ok=True)
    failed = []
    for item in a.only:
        try:
            {'hairdeck': get_hairdeck, 'birefnet': get_birefnet, 'sam3': get_sam3, 'dinov3': get_dinov3,
             'blender': get_blender, 'test_data': get_test_data, 'charm_test_pc': get_charm_test_pc,
             'dora': get_dora}.get(item, lambda r, x: get_source(r, x, item))(root, a)
        except Exception as e:  # noqa: BLE001  (report every item, then fail)
            log('FAILED', item, f'{type(e).__name__}: {e}')
            failed.append(item)
    if failed:
        sys.exit(f'not ready: {failed}')
    log('ALL_READY', root)


if __name__ == '__main__':
    main()
