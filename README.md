# CHLA 遥感图像反演系统（以下教程仅限于 Linux 系统）

这是一个基于 Flask + Python 遥感算法脚本的叶绿素 a（Chla）反演系统。
前端负责上传影像、选择算法和查看结果，后端负责调用算法脚本、保存结果并记录历史数据。

## 1. 从 GitHub 拉代码到本机

如果你是第一次在本机使用这个项目，先克隆仓库：

```bash
git clone https://github.com/CoderRenZhe/chla_system.git
cd chla_system
```

如果你已经把项目放在别的位置，例如 `/root/chla_system`，那就直接进入项目目录：

```bash
cd /root/chla_system
```

## 2. 创建 Python 虚拟环境

### Conda 太慢了，用 pip（五分钟配好）

推荐直接使用 `venv`：

```bash
cd /root/chla_system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

安装完成后，可以用下面这条命令验证依赖是否正常：

```bash
python -c "import numpy, pandas, matplotlib, sklearn, rasterio, xgboost; print('ok')"
```

如果输出 `ok`，说明核心依赖已经安装成功。

## 3. 启动项目

前端页面和后端接口都由 Flask 服务提供，启动方式如下：

```bash
cd /root/chla_system
source .venv/bin/activate
python backend/app.py
```

启动成功后，浏览器打开：

```text
http://127.0.0.1:5000/chla
```

## 4. 使用流程

项目的基本使用顺序如下：

1. 打开页面 `http://127.0.0.1:5000/chla`
2. 上传遥感影像文件，格式支持 `.tif` 或 `.tiff`
3. 选择反演算法，例如随机森林、支持向量机或 XGBoost
4. 点击“运行反演”
5. 查看结果图、评价指标以及历史记录
6. 如有需要，可以下载反演输出结果

## 5. 项目目录解释

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
├── README.md 项目说明文档
├── .gitignore Git 忽略规则
└── requirements.txt 环境依赖文件，给 pip 用
```

## 6. 关键文件说明

- `backend/app.py`
  负责提供 Flask 接口、处理文件上传、调用算法脚本、保存历史记录。
- `scripts/train_rf.py`
  使用随机森林做训练和反演。
- `scripts/train_svr.py`
  使用支持向量机做训练和反演。
- `scripts/train_xgb.py`
  使用 XGBoost 做训练和反演。
- `web/chla_console.html`
  前端页面文件，页面入口由后端 `/chla` 路由提供。

## 7. 数据和输出文件说明

- 原始采样表：`data/raw/sampling-sites.csv`
- 默认遥感影像：`data/raw/Sentinel2.tif`
- 上传后的原图和预览图：`data/raw/uploads/`
- 历史记录数据库：`data/chla_system.db`
- 算法输出结果目录：`data/output/rf/`、`data/output/svr/`、`data/output/xgb/`

## 8. 补充说明

- 后端默认监听地址为 `0.0.0.0:5000`
- 主页面访问路径为 `/chla`
- 历史记录时间在页面中按北京时间显示
- 如果你要把项目发给别人，通常不需要提交 `.venv`、`__pycache__`、`data/output/`、`data/chla_system.db` 这类本地运行产物
