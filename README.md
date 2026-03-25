# CHLA 遥感图像反演系统

这是一个基于 Flask + Python 遥感算法脚本的叶绿素 a（Chla）反演系统。
前端负责上传影像、选择算法和查看结果，后端负责调用算法脚本并记录历史数据。

## 快速安装

### Conda 太慢了，用 pip（五分钟配好）

新建一个 venv，然后用 pip：

```bash
cd /root/chla_system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

然后验证：

```bash
python -c "import numpy, pandas, matplotlib, sklearn, rasterio, xgboost; print('ok')"
```

## 项目目录解释

```bash
~/chla_system/
├── data/ 存放数据
│   ├── chla_system.db 数据库存放历史数据
│   ├── raw/ 存放原始数据
│   │   ├── uploads/ 上传的 tif 原图和转换的 png 图
│   │   ├── Sentinel2.tif 遥感影像原图
│   │   └── sampling-sites.csv 实测采样点样本表
│   └── output/ 放每种算法跑完后的结果
│       ├── rf/
│       ├── svr/
│       └── xgb/
├── scripts/ 放核心算法脚本
│   ├── train_rf.py 读取采样点数据、训练随机森林、参数调优等等
│   ├── train_svr.py 同上
│   └── train_xgb.py 同上
├── web/ 前端页面
│   └── chla_console.html
├── backend/ 放后端代码
│   └── app.py 中间人，负责连接前端和算法脚本
└── requirements.txt 环境依赖文件，给 pip 用
```

## 运行方式

写好前端后的运行方式：

```bash
cd /root/chla_system
source .venv/bin/activate
python backend/app.py
```

然后浏览器打开：`http://127.0.0.1:5000/chla`

## 补充说明

- 后端默认监听地址为 `0.0.0.0:5000`
- 主页面访问路径为 `/chla`
- 历史记录保存在 `data/chla_system.db`
- 上传影像与预览图保存在 `data/raw/uploads/`
- 反演输出结果保存在 `data/output/` 对应算法子目录下
