# CHLA 遥感图像反演系统 Windows 使用说明

这份文档专门给 Windows 用户使用。
如果你是在 Windows 上运行这个项目，优先看这份说明，不要直接照搬 Linux 里的 `bin/activate` 写法。

## 1. 从 GitHub 拉代码到本机

打开 PowerShell 或 CMD，先克隆仓库：

```powershell
git clone https://github.com/CoderRenZhe/chla_system.git
cd chla_system
```

如果你已经把项目解压或放到了别的目录，例如：

```text
D:\projects\chla_system
```

那就直接进入这个项目目录。

## 2. 创建 Windows 虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
```

创建完成后，Windows 下虚拟环境目录通常是：

```text
.venv\Scripts\
```

不是 Linux/macOS 的：

```text
.venv/bin/
```

## 3. 激活虚拟环境

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

如果遇到执行策略报错，可以先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

然后再执行：

```powershell
.venv\Scripts\Activate.ps1
```

### CMD

```cmd
.venv\Scripts\activate.bat
```

## 4. 安装依赖

激活虚拟环境后执行：

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 5. 检查依赖是否安装成功

安装完以后，先不要急着开网页，先检查关键依赖：

```powershell
python -c "import flask, numpy, pandas, matplotlib, sklearn, rasterio, xgboost; print('ok')"
```

如果输出：

```text
ok
```

说明核心依赖已经安装成功。

如果这里直接报错，优先说明环境没装完整，而不是网页本身有问题。

## 6. 启动项目

在项目根目录执行：

```powershell
python backend/app.py
```

启动后，浏览器打开：

```text
http://127.0.0.1:5000/chla
```

## 7. 正常使用流程

1. 打开 `http://127.0.0.1:5000/chla`
2. 上传 `.tif` 或 `.tiff` 文件
3. 选择反演算法
4. 点击“运行反演”
5. 等待结果图和指标出现

## 8. Windows 下最容易踩的坑

### 1. 把 Linux 命令照抄到 Windows

Windows 不要用：

```bash
source .venv/bin/activate
```

Windows 应该用：

```powershell
.venv\Scripts\Activate.ps1
```

或者：

```cmd
.venv\Scripts\activate.bat
```

### 2. 网页能打开，不代表算法环境没问题

这个项目里：
- 网页能打开，只能说明 Flask 相关部分大概率正常
- 点“运行反演”时，后端还会继续调用算法脚本

所以会出现这种情况：
- 页面正常打开
- tif 也能上传
- 但一运行算法就失败

这通常是因为这些包在 Windows 上没装好：
- `xgboost`
- `rasterio`
- `scikit-learn`

### 3. XGBoost 报“算法运行失败”

如果页面提示 XGBoost 运行失败，优先执行：

```powershell
python -c "import xgboost; print('xgboost ok')"
```

如果这里报错，说明就是 XGBoost 环境有问题。

### 4. 单独运行脚本比看网页更容易定位问题

如果网页点击“运行反演”后没反应，直接在终端单独跑：

```powershell
python scripts\train_xgb.py
```

或者：

```powershell
python scripts\train_rf.py
python scripts\train_svr.py
```

这样可以直接看到真实报错，比只看网页更容易定位。

## 9. 检查当前 Python 是否来自虚拟环境

激活 `.venv` 后执行：

```powershell
python -c "import sys; print(sys.executable)"
```

输出应该类似：

```text
D:\...\chla_system\.venv\Scripts\python.exe
```

如果不是 `.venv` 里的 Python，说明你并没有真正用项目自己的虚拟环境在运行。

## 10. 检查关键数据文件是否存在

在项目根目录执行：

```powershell
dir data\raw
```

至少应该能看到这些关键文件：
- `sampling-sites.csv`
- `Sentinel2.tif`（如果你要用默认数据）

如果文件缺失，也会导致算法无法正常运行。

## 11. 项目目录解释

```text
chla_system/
├── data/
│   ├── chla_system.db
│   ├── raw/
│   │   ├── uploads/
│   │   ├── Sentinel2.tif
│   │   └── sampling-sites.csv
│   └── output/
│       ├── rf/
│       ├── svr/
│       └── xgb/
├── scripts/
│   ├── train_rf.py
│   ├── train_svr.py
│   └── train_xgb.py
├── web/
│   └── chla_console.html
├── backend/
│   └── app.py
├── README.md
├── README_WINDOWS.md
└── requirements.txt
```

## 12. 一句话判断思路

如果是 Windows 用户，出现“网页能打开，但运行反演没反应”时，优先按这个顺序查：

1. 有没有正确激活 `.venv\Scripts\...`
2. `python -c "import ..."` 能不能通过
3. `python scripts\train_xgb.py` 能不能单独跑
4. `data\raw` 里的关键文件在不在

这四步通常比反复点网页更有效。
