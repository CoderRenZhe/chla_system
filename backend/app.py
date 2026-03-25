import json
import shutil
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
UPLOAD_DIR = DATA_RAW_DIR / 'uploads'
OUTPUT_DIR = DATA_DIR / 'output'
CSV_PATH = DATA_RAW_DIR / 'sampling-sites.csv'
DB_PATH = DATA_DIR / 'chla_system.db'

ALGORITHM_CONFIG = {
    'rf': {
        'script': PROJECT_ROOT / 'scripts' / 'train_rf.py',
        'output_subdir': 'rf',
        'name': '随机森林',
    },
    'svr': {
        'script': PROJECT_ROOT / 'scripts' / 'train_svr.py',
        'output_subdir': 'svr',
        'name': '支持向量机',
    },
    'xgb': {
        'script': PROJECT_ROOT / 'scripts' / 'train_xgb.py',
        'output_subdir': 'xgb',
        'name': 'XGBoost',
    },
}
ALLOWED_EXTENSIONS = {'.tif', '.tiff'}

app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS inversion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                algorithm TEXT NOT NULL,
                algorithm_name TEXT NOT NULL,
                input_image TEXT NOT NULL,
                preview_image TEXT,
                result_image TEXT NOT NULL,
                result_tif TEXT NOT NULL,
                scatter_plot TEXT,
                rmse REAL NOT NULL,
                mae REAL NOT NULL,
                r2 REAL NOT NULL,
                re REAL NOT NULL,
                best_params TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        conn.commit()


def build_file_url(path: Path) -> str:
    relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return f'/api/files/{relative_path.as_posix()}'


def to_project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    valid = np.isfinite(band)
    if not np.any(valid):
        return np.zeros_like(band, dtype=np.uint8)
    low, high = np.percentile(band[valid], [2, 98])
    if high <= low:
        scaled = np.clip(band, 0, 1)
    else:
        scaled = np.clip((band - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def create_preview_png(tif_path: Path, png_path: Path) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            red = normalize_band(src.read(min(3, src.count)))
            green = normalize_band(src.read(min(2, src.count)))
            blue = normalize_band(src.read(1))
            preview = np.dstack([red, green, blue])
        else:
            gray = normalize_band(src.read(1))
            preview = np.dstack([gray, gray, gray])

    plt.figure(figsize=(6, 6))
    plt.imshow(preview)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    return png_path


def save_uploaded_tif(image_file) -> tuple[str, Path, Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    suffix = Path(image_file.filename).suffix.lower()
    safe_name = secure_filename(image_file.filename) or f'input{suffix}'
    upload_path = UPLOAD_DIR / f'{run_id}_{safe_name}'
    preview_path = UPLOAD_DIR / f'{run_id}_preview.png'

    try:
        image_file.save(upload_path)
        create_preview_png(upload_path, preview_path)
    finally:
        # Windows may keep uploaded files locked a bit longer unless the request-side
        # file handle is explicitly closed before the training subprocess reads it.
        image_file.close()

    return run_id, upload_path, preview_path


def save_history_record(run_id: str, algorithm: str, algorithm_name: str, upload_path: Path, preview_path: Path, metrics: dict) -> None:
    with get_db_connection() as conn:
        conn.execute(
            '''
            INSERT OR REPLACE INTO inversion_history (
                run_id, algorithm, algorithm_name, input_image, preview_image,
                result_image, result_tif, scatter_plot,
                rmse, mae, r2, re, best_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                run_id,
                algorithm,
                algorithm_name,
                to_project_relative(upload_path),
                to_project_relative(preview_path),
                metrics['result_map'],
                metrics['result_tif'],
                metrics.get('scatter_plot'),
                float(metrics['rmse']),
                float(metrics['mae']),
                float(metrics['r2']),
                float(metrics['re']),
                json.dumps(metrics.get('best_params', {}), ensure_ascii=False),
            ),
        )
        conn.commit()


def fetch_history_records() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            '''
            SELECT id, run_id, algorithm, algorithm_name, input_image, preview_image,
                   result_image, result_tif, scatter_plot,
                   rmse, mae, r2, re, best_params, created_at
            FROM inversion_history
            ORDER BY datetime(created_at) DESC, id DESC
            '''
        ).fetchall()

    records = []
    for row in rows:
        record = dict(row)
        record['best_params'] = json.loads(record['best_params']) if record['best_params'] else {}
        record['preview_image_url'] = build_file_url(PROJECT_ROOT / record['preview_image']) if record['preview_image'] else ''
        record['result_image_url'] = build_file_url(PROJECT_ROOT / record['result_image'])
        record['result_tif_url'] = build_file_url(PROJECT_ROOT / record['result_tif'])
        record['scatter_plot_url'] = build_file_url(PROJECT_ROOT / record['scatter_plot']) if record['scatter_plot'] else ''
        records.append(record)
    return records


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'project_root': str(PROJECT_ROOT)})


@app.get('/')
def index():
    return jsonify({
        'message': 'CHLA backend is running.',
        'web_entry': str(PROJECT_ROOT / 'web' / 'chla_console.html'),
    })


@app.get('/chla')
def demo():
    return send_file(PROJECT_ROOT / 'web' / 'chla_console.html')


@app.get('/api/files/<path:relative_path>')
def get_project_file(relative_path: str):
    file_path = (PROJECT_ROOT / relative_path).resolve()
    if not file_path.exists() or (PROJECT_ROOT.resolve() not in file_path.parents and file_path != PROJECT_ROOT.resolve()):
        return jsonify({'error': '文件不存在'}), 404
    return send_file(file_path)


@app.get('/api/history')
def history():
    return jsonify({'status': 'success', 'items': fetch_history_records()})


@app.post('/api/upload-preview')
def upload_preview():
    image_file = request.files.get('image')
    if image_file is None or image_file.filename == '':
        return jsonify({'error': '请上传遥感影像文件'}), 400

    suffix = Path(image_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({'error': '仅支持上传 .tif 或 .tiff 文件'}), 400

    try:
        run_id, upload_path, preview_path = save_uploaded_tif(image_file)
    except Exception as exc:
        return jsonify({'error': f'预览图生成失败: {exc}'}), 500

    return jsonify({
        'status': 'success',
        'run_id': run_id,
        'uploaded_image': to_project_relative(upload_path),
        'preview_image_url': build_file_url(preview_path),
    })


@app.post('/api/invert')
def invert():
    algorithm = request.form.get('algorithm', '').strip().lower()
    image_file = request.files.get('image')

    if algorithm not in ALGORITHM_CONFIG:
        return jsonify({'error': '无效的算法类型'}), 400
    if image_file is None or image_file.filename == '':
        return jsonify({'error': '请上传遥感影像文件'}), 400

    suffix = Path(image_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({'error': '仅支持上传 .tif 或 .tiff 文件'}), 400
    if not CSV_PATH.exists():
        return jsonify({'error': '采样点 CSV 文件不存在'}), 500

    try:
        run_id, upload_path, preview_path = save_uploaded_tif(image_file)
    except Exception as exc:
        return jsonify({'error': f'上传文件处理失败: {exc}'}), 500

    config = ALGORITHM_CONFIG[algorithm]
    run_output_dir = OUTPUT_DIR / config['output_subdir'] / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(config['script']),
        '--csv', str(CSV_PATH),
        '--image', str(upload_path),
        '--output-dir', str(run_output_dir),
    ]

    try:
        print(f'[invert] running command: {command}')
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print('[invert] stdout:')
        print(result.stdout)
        print('[invert] stderr:')
        print(result.stderr)
    except subprocess.CalledProcessError as exc:
        print('[invert] subprocess failed')
        print(f'[invert] command: {command}')
        print('[invert] stdout:')
        print(exc.stdout)
        print('[invert] stderr:')
        print(exc.stderr)
        return jsonify({
            'error': '算法运行失败',
            'algorithm': algorithm,
            'stdout': exc.stdout,
            'stderr': exc.stderr,
        }), 500
    except Exception as exc:
        import traceback
        print('[invert] unexpected error:')
        traceback.print_exc()
        return jsonify({
            'error': f'后端异常: {exc}'
        }), 500

    metrics_path = run_output_dir / 'metrics.json'
    if not metrics_path.exists():
        return jsonify({
            'error': '算法运行完成，但未生成 metrics.json',
            'stdout': result.stdout,
            'stderr': result.stderr,
        }), 500

    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    metrics['result_map'] = to_project_relative(Path(metrics['result_map']))
    metrics['result_tif'] = to_project_relative(Path(metrics['result_tif']))
    metrics['scatter_plot'] = to_project_relative(Path(metrics['scatter_plot'])) if metrics.get('scatter_plot') else ''

    save_history_record(run_id, algorithm, metrics['algorithm_name'], upload_path, preview_path, metrics)

    response = {
        'status': 'success',
        'run_id': run_id,
        'algorithm': algorithm,
        'algorithm_name': metrics['algorithm_name'],
        'metrics': {
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            're': metrics['re'],
        },
        'best_params': metrics.get('best_params', {}),
        'preview_image_url': build_file_url(preview_path),
        'result_image_url': build_file_url(PROJECT_ROOT / metrics['result_map']),
        'result_tif_url': build_file_url(PROJECT_ROOT / metrics['result_tif']),
        'scatter_plot_url': build_file_url(PROJECT_ROOT / metrics['scatter_plot']) if metrics['scatter_plot'] else '',
        'uploaded_image': to_project_relative(upload_path),
        'stdout': result.stdout,
    }
    return jsonify(response)


@app.post('/api/reset-default-input')
def reset_default_input():
    latest_file = request.json.get('source') if request.is_json else None
    if not latest_file:
        return jsonify({'error': '缺少 source 参数'}), 400

    source_path = (PROJECT_ROOT / latest_file).resolve()
    if not source_path.exists():
        return jsonify({'error': '源文件不存在'}), 404

    shutil.copy2(source_path, DATA_RAW_DIR / 'Sentinel2.tif')
    return jsonify({'status': 'success'})


init_db()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
