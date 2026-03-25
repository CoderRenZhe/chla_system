import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'xgb'

CSV_PATH = DATA_RAW_DIR / 'sampling-sites.csv'
IMG_PATH = DATA_RAW_DIR / 'Sentinel2.tif'


def parse_args():
    parser = argparse.ArgumentParser(description='Train XGBoost model and invert chlorophyll-a from remote sensing imagery.')
    parser.add_argument('--csv', type=Path, default=CSV_PATH, help='Path to sampling CSV file.')
    parser.add_argument('--image', type=Path, default=IMG_PATH, help='Path to input Sentinel-2 GeoTIFF file.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory for output files.')
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv
    img_path = args.image
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_tif_path = output_dir / 'xgb_result.tif'
    scatter_plot_path = output_dir / 'xgb_scatter.png'
    result_map_path = output_dir / 'xgb_result_map.png'
    metrics_path = output_dir / 'metrics.json'

    print('>>> 步骤 1: 读取并预处理 CSV 数据...')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f'错误: 找不到文件 {csv_path}')
        return

    feature_cols = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    target_col = 'Chla'

    if target_col not in df.columns:
        possible_cols = [c for c in df.columns if 'chla' in c.lower() or '叶绿素' in c]
        if possible_cols:
            target_col = possible_cols[0]
        else:
            raise ValueError('CSV中未找到目标列')

    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    print(f'    有效样本数: {len(df_clean)}')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print('\n>>> 步骤 2: 构建并优化 XGBoost 模型...')
    xgb_model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
    param_grid = {
        'n_estimators': [500],
        'max_depth': [3],
        'learning_rate': [0.01],
        'subsample': [0.8],
        'colsample_bytree': [0.8, 1.0],
    }

    print('    正在配置模型 (使用之前的最佳参数加速)...')
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    print(f'    最佳参数: {grid_search.best_params_}')

    print('\n>>> 步骤 3: 模型评估...')
    y_pred = best_xgb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    re = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f'    测试集 R2 : {r2:.4f}')
    print(f'    测试集 RMSE : {rmse:.4f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='green', alpha=0.6, label='Samples')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    plt.xlabel(f'Measured {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'XGBoost Model Evaluation\nR2={r2:.3f}, RMSE={rmse:.3f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(scatter_plot_path, dpi=300)
    print(f'    [成功] 散点图已保存至: {scatter_plot_path}')
    plt.close()

    print('\n>>> 步骤 4: 影像反演...')
    with rasterio.open(img_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        img_data = src.read(indexes=[1, 2, 3, 4, 5, 6, 7])
        img_reshaped = img_data.reshape(7, -1).T
        valid_mask = np.all(img_reshaped > 0, axis=1) & np.all(~np.isnan(img_reshaped), axis=1)

        pred_result = np.full(img_reshaped.shape[0], np.nan, dtype=np.float32)
        if np.sum(valid_mask) > 0:
            pred_result[valid_mask] = best_xgb.predict(img_reshaped[valid_mask])

        final_image = pred_result.reshape(height, width)
        meta.update({'count': 1, 'dtype': 'float32', 'nodata': np.nan})
        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(final_image, 1)
            print(f'    [成功] 反演结果TIF已保存至: {output_tif_path}')

    print('\n>>> 步骤 5: 结果可视化...')
    plt.figure(figsize=(10, 8))
    plt.imshow(final_image, cmap='viridis')
    plt.colorbar(label='Chla (mg/m^3)')
    plt.title('XGBoost Inversion Result')
    plt.axis('off')
    plt.savefig(result_map_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    print(f'    [成功] 结果效果图已保存至: {result_map_path}')
    plt.close()

    metrics = {
        'algorithm': 'xgb',
        'algorithm_name': 'XGBoost',
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        're': float(re),
        'best_params': grid_search.best_params_,
        'input_image': str(img_path),
        'input_csv': str(csv_path),
        'result_tif': str(output_tif_path),
        'result_map': str(result_map_path),
        'scatter_plot': str(scatter_plot_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'    指标文件已保存至: {metrics_path}')

    print('\n>>> 全部完成！请去文件夹查看图片。')


if __name__ == '__main__':
    main()
