"""Regenerate architecture DOTs from this repository, optionally render via Graphviz.

02 and 05 are AST-derived imports; 08 and 09 use a saved Graft MCP snapshot.
Other diagrams are explicitly reviewed models, not inferred call graphs.
No hardware modules are imported by this script.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/graft_visualizations"
SNAPSHOT = OUT / "graft-analysis.json"


def quote(value):
    return json.dumps(value)


def graph(title, nodes, edges, direction="LR", packed=False):
    lines = ['digraph architecture {', f'graph [rankdir={direction}, bgcolor="#f8fafc", pad=0.4, nodesep=0.4, ranksep=0.8, label={quote(title)}, labelloc=t, fontname="DejaVu Sans", fontsize=20];', 'node [shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7", fontname="DejaVu Sans", fontsize=11];', 'edge [color="#475569", fontname="DejaVu Sans", fontsize=9];']
    if packed:
        lines.append('graph [pack=true, packmode="array_u3"];')
    for name, label in nodes.items():
        lines.append(f'{quote(name)} [label={quote(label)}];')
    for source, target, label in sorted(set(edges)):
        lines.append(f'{quote(source)} -> {quote(target)} [label={quote(label)}];')
    return '\n'.join(lines + ['}', ''])


def modules():
    paths = [ROOT / "main_ui.py"]
    for directory in ("core", "ui", "tests"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    result = {}
    for path in paths:
        name = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
        if name.endswith('.__init__'):
            name = name[:-9]
        result[name] = path
    return result


def import_edges(sources):
    edges = []
    for source, path in sources.items():
        package = source if path.name == "__init__.py" else source.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split('.')
                    prefix = parts[:len(parts) - node.level + 1]
                    target = '.'.join(prefix + ([node.module] if node.module else []))
                else:
                    target = node.module or ''
                targets = [target]
                # `from package import module` is a module dependency when present.
                targets += [target + '.' + alias.name for alias in node.names if target + '.' + alias.name in sources]
            else:
                continue
            edges.extend((source, target, "import") for target in targets if target in sources and source != target)
    return edges


def graft_diagrams(sources):
    """Render recorded MCP evidence; reject stale or incomplete source maps."""
    if not SNAPSHOT.exists():
        raise ValueError(f'Missing Graft snapshot: {SNAPSHOT}')
    snapshot = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    files = {}
    for result in snapshot['maps']:
        if result.get('truncated'):
            raise ValueError('Graft map is truncated; collect narrower scopes first')
        for file in result['files']:
            files[file['path']] = file
    current = {str(path.relative_to(ROOT)) for path in sources.values()}
    if current != set(files):
        raise ValueError('Graft file inventory is stale; refresh graft-analysis.json')
    for name in sorted(current):
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if snapshot['source_sha256'].get(name) != digest:
            raise ValueError(f'Graft source snapshot is stale: {name}')

    stamp = snapshot['captured_at']
    # File membership and symbols come directly from graft_map. Keep methods
    # in the evidence, but show top-level classes/functions for legibility.
    nodes, edges = {}, []
    for name, file in sorted(files.items()):
        folder = str(Path(name).parent)
        group = 'directory:' + folder
        nodes[group] = folder if folder != '.' else 'Application entry point'
        symbols = file['symbols']
        top = [s['name'] for s in symbols
               if s['kind'] in ('class', 'function') and s.get('startLine')]
        shown = top[:6]
        if len(top) > len(shown):
            shown.append(f'... +{len(top) - len(shown)} classes/functions')
        nodes[name] = '\n'.join([Path(name).name, f'{len(symbols)} Graft symbols'] + shown)
        edges.append((group, name, 'contains'))
    inventory = graph(
        f'Graft working-tree symbol inventory | {len(files)} files\n{stamp} | containment only',
        nodes, edges, packed=True)

    diff = snapshot['structural_diff']
    groups = {}
    for file in diff['files']:
        scope = file['path'].split('/')[0] if '/' in file['path'] else 'root'
        key = (file['status'], scope)
        changes = file['diff']
        counts = ' '.join(f'{mark}{len(changes[k])}' for mark, k in
                          (('+', 'added'), ('-', 'removed'), ('~', 'changed'))
                         if changes[k])
        groups.setdefault(key, []).append(file['path'] + (f' [{counts}]' if counts else ''))
    for name in snapshot['untracked_python_files']:
        groups.setdefault(('untracked Python', name.split('/')[0]), []).append(name)
    nodes, edges = {}, []
    for (status, scope), names in sorted(groups.items()):
        origin = 'git ls-files' if status == 'untracked Python' else 'graft_diff'
        nodes[status] = f'{status}\nsource: {origin}'
        key = status + ':' + scope
        nodes[key] = '\n'.join([f'{scope} ({len(names)} files)'] + sorted(names))
        edges.append((status, key, 'files'))
    changes = graph(
        f'Working-tree change snapshot vs HEAD {snapshot["head_commit"][:8]}\n'
        f'{stamp} | +/-/~ = top-level symbol changes; untracked Python supplemented by Git',
        nodes, edges)
    return {'08-graft-symbol-map': inventory, '09-graft-working-tree': changes}


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = modules()
    imports = import_edges(sources)
    active = {name: path for name, path in sources.items() if not name.startswith('tests')}
    used = {n for a, b, _ in imports if a in active and b in active for n in (a, b)}
    diagrams = {}
    diagrams.update(graft_diagrams(sources))
    diagrams['02-module-dependencies'] = graph(
        'Static imports only - caller -> imported module\nAST-derived; NOT runtime calls or ownership',
        {n: n.replace('core.calibration.', 'calibration.').replace('sequences.', 'sequences.\n') for n in sorted(used)},
        [(a, b, label) for a, b, label in imports if a in active and b in active])
    test_edges = [(a, b, label) for a, b, label in imports if a.startswith('tests.') and not b.startswith('tests')]
    diagrams['05-test-coverage-map'] = graph('Test import map - NOT measured execution coverage',
        {n: n for a, b, _ in test_edges for n in (a, b)}, test_edges)
    diagrams['01-project-overview'] = graph('Repository responsibilities - reviewed architecture', {
        'main': 'main_ui.py\nwidgets / display / input', 'ui': 'ui/\ncore_bridge.py + wizard_widget.py',
        'core': 'core/calibration/calibration_core.py\nexecution and device lifetime owner',
        'seq': 'core/calibration/sequences/\nmarker / collection / Step 1 / 1.5 / 2 / full / result',
        'cal': 'core/calibration/\nBase + Marker + Joint + HeadCamera + Intrinsics\ncalibration_optimizer.py (flat)',
        'robot': 'core/robot/\nrobot_core.py / motion.py / home_offset.py',
        'marker': 'core/marker_detection.py\nexisting detection / transforms / simulation',
        'camera': 'core/camera_processing.py\nRealSenseCamera + latest-frame cache',
        'storage': 'core/storage.py\npaths / configs / datasets / results / artifacts',
        'config': 'config/ + result/\nexisting file formats retained'},
        [('main','ui','commands / results'), ('ui','core','run / cancel / snapshot'), ('core','seq','dispatch'),
         ('seq','cal','reuse'), ('core','robot','control'), ('core','marker','create via ObservationSource'),
         ('marker','camera','camera_factory'), ('core','storage','I/O'), ('cal','storage','I/O'),
         ('robot','storage','I/O'), ('storage','config','read / save')])
    diagrams['03-calibration-classes'] = graph('Inheritance and ownership - labels define the relationship', {
        'core': 'CalibrationCore', 'base': 'BaseCalibrator', 'ops': 'RobotOperations',
        'marker': 'MarkerCalibrator', 'joint': 'JointCalibrator', 'head': 'HeadCameraCalibrator',
        'intr': 'IntrinsicsCalibrator', 'obs': 'ObservationSource', 'engine': 'Marker_Transform',
        'cam': 'RealSenseCamera', 'full': 'run_full', 'step': 'Step 1 / 1.5 / 2',
        'opt': 'QPCalibrationOptimizer / CalibrationOptimizer'},
        [('base','ops','inherits'), ('marker','base','inherits'), ('joint','base','inherits'), ('head','base','inherits'),
         ('core','marker','owns'), ('core','joint','owns'), ('core','head','owns'), ('core','intr','owns'),
         ('core','obs','owns'), ('obs','engine','wraps; serialized detection'), ('engine','cam','owns device'),
         ('base','obs','injected marker_st reference (NOT flag)'), ('core','full','dispatch'),
         ('full','step','same step implementations'), ('step','opt','Step 2 solves')], 'TB')
    diagrams['04-runtime-flow'] = graph('Runtime / data flow - reviewed call sites, NOT static imports', {
        'ui': 'UI main thread', 'worker': 'SequenceWorker (QThread)', 'core': 'CalibrationCore\none active sequence; stop event',
        'full': 'Full: Step 1 -> Step 1.5 -> Step 2', 'seq': 'standalone step / sequence',
        'obs': 'ObservationSource\nserialized existing marker engine', 'camera': 'camera-capture thread\nonly pipeline reader; temperature every 5s',
        'cache': 'locked latest-frame cache\nimage / frame ID / timestamp / temperature',
        'res': 'SequenceResult\ncompleted / cancelled / failed\ncompleted steps + partial observations / iterations',
        'bridge': 'CoreBridge queued Qt signals'},
        [('ui','worker','start command'), ('worker','core','run'), ('ui','core','cancel'), ('core','full','full'),
         ('core','seq','single step'), ('full','seq','reuse; stop prevents next step'),
         ('seq','obs','request marker sample'), ('obs','cache','wait for fresh frame; copy'),
         ('camera','cache','publish'), ('ui','cache','via core.get_monitor_snapshot(); COPY ONLY'),
         ('seq','res','record progress even on stop'), ('res','bridge','result'), ('bridge','ui','render on GUI thread')])
    diagrams['06-working-tree-changes'] = graph('This refactor migration map - NOT a live Git diff', {
        'oldworker': 'main_ui.py worker / orchestration code', 'bridge': 'ui/core_bridge.py',
        'core': 'calibration/calibration_core.py', 'oldfull': 'calibration/FullAutoSequence.py', 'steps': 'calibration/sequences/',
        'oldcam': 'marker_detection.RealSenseCamera', 'cam': 'camera_processing.RealSenseCamera',
        'oldrobot': 'robot_motion.py + calibration/homeoffset_core.py', 'robot': 'robot/motion.py + robot/home_offset.py',
        'oldstorage': 'paths.py + scattered I/O', 'storage': 'storage.py (one file)',
        'oldwizard': 'core/wizard_widget.py', 'wizard': 'ui/wizard_widget.py'},
        [('oldworker','bridge','Qt adapters'), ('oldworker','core','execution owner'), ('oldfull','steps','step1 + reusable full'),
         ('oldcam','cam','extract camera only'), ('oldrobot','robot','group robot responsibilities'),
         ('oldstorage','storage','group into classes'), ('oldwizard','wizard','UI belongs in UI')])
    diagrams['07-graft-workflow'] = graph('Graft analysis and reproducible diagrams - current workspace workflow', {
        'workspace': 'camera_ws\nauthorized workspace',
        'route': 'Explicit cwd: /home/jsm/camera_ws',
        'map': 'graft_map\nsmall directory / file scopes',
        'refs': 'code_refs\nexisting file scopes; text fallback',
        'diff': 'graft_diff\nHEAD vs working tree; tracked files',
        'git': 'git ls-files\nuntracked Python supplement',
        'snapshot': 'graft-analysis.json\nMCP results + receipts + source hashes',
        'ast': 'Python AST\nstatic imports',
        'models': 'Reviewed architecture models\nownership / flow / migration',
        'script': 'scripts/generate_architecture.py\nvalidate snapshot + generate DOT',
        'render': 'Graphviz\nSVG + PNG'},
        [('workspace','route','scope'), ('route','map','parse'), ('route','refs','inspect'),
         ('route','diff','compare'), ('map','snapshot','symbols'), ('refs','snapshot','evidence'),
         ('diff','snapshot','changes'), ('git','snapshot','new files'),
         ('snapshot','script','08 / 09'), ('ast','script','02 / 05'),
         ('models','script','01 / 03 / 04 / 06 / 07'), ('script','render','01-09')])
    for name, source in diagrams.items():
        (OUT / (name + '.dot')).write_text(source, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    generate()
    if args.render:
        for path in sorted(OUT.glob('*.dot')):
            for fmt in ('svg', 'png'):
                subprocess.run(['dot', '-T' + fmt, str(path), '-o', str(path.with_suffix('.' + fmt))], check=True)
    print('Updated', OUT)
