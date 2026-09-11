# HkcTracker · Bubble Analyzer

面向气泡图像与视频的桌面分析工具，使用 **PySide6 / Qt、Ultralytics YOLO 实例分割和 BoT-SORT / ByteTrack**，提供气泡跟踪、运动测量与基于图像的气含率分析。

当前主入口为 `main.py`，启动后包含两个独立工作流：

| 工作流 | 输入 | 主要功能 | 输出 |
| --- | --- | --- | --- |
| **Bubble Tracking** | 视频或图像序列 | 实例分割、跨帧 ID、轨迹、速度、面积、等效直径与上升运动分析 | 标注 MP4、运动 CSV、轨迹 TXT |
| **Gas Holdup** | 图像文件夹 | ROI 内掩膜并集面积、气泡数量、二维面积占比、平均等效直径 | 每帧覆盖图 PNG、统计 CSV |

两者共用 YOLO 推理和图像读取模块；Gas Holdup 不依赖跟踪 ID、轨迹寿命或运动过滤结果。

## 快速开始

### 1. 准备 Python 环境

下载或克隆本仓库后，在项目根目录运行。推荐使用 Conda 和 Python 3.10；当前本地开发环境名为 `hkc`。

```powershell
conda create -n hkc python=3.10 -y
conda activate hkc
python -m pip install -r requirements.txt
```

已有 `hkc` 环境时，跳过创建步骤。主要依赖包括 PyTorch、Ultralytics、OpenCV、NumPy、SciPy、PySide6、`lap` 和 `cython-bbox`，完整列表见 [requirements.txt](requirements.txt)。

依赖暂未锁定版本。需要 NVIDIA GPU 加速时，请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与驱动兼容的 PyTorch，并确认 CUDA 可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Bubble Tracking 的 Device 可选择 `auto`、`cpu` 或 `cuda:0`；Gas Holdup 当前使用自动设备选择。显式选择 CUDA 而 PyTorch 无法使用 GPU 时，程序会报错。

### 2. 准备模型和输入数据

- **模型权重不随 Git 仓库分发。** 请自行准备与本项目兼容、针对气泡训练的 YOLO 实例分割权重。
- 默认模型路径为项目根目录的 `3900.pt`，也可在界面的 **Model** 中选择其他路径。
- 模型选择框支持 `.pt` 和 `.onnx`；ONNX 需要兼容的导出模型及相应运行时，建议先用 `.pt` 跑通。
- Gas Holdup 必须使用实例分割模型。普通目标检测模型只有检测框，不能用于该流程的掩膜面积计算。
- `weights/yolo11n.pt` 等通用检测权重不能替代经过气泡训练的分割模型。

仓库保留 `data/dataset/test_images/` 下的三张 TIFF 图片，可用于分割与 Gas Holdup 试运行。视频放在本地 `data/videos/` 或仓库外，再从界面选择。

支持的图像格式：`.jpg`、`.jpeg`、`.webp`、`.bmp`、`.png`、`.tif`、`.tiff`。文件夹会递归读取，并按文件名自然排序，例如 `frame_2` 排在 `frame_10` 前。运动测量应使用按时间命名、间隔已知的连续序列；文件名编号的差值不会自动转换为采样时间。

### 3. 启动界面

```powershell
conda run --no-capture-output -n hkc python main.py
```

已激活环境时也可直接执行：

```powershell
python main.py
```

Windows 的 `START.bat` 当前写有本机 Conda 路径 `C:\Fluids\miniconda3\condabin\conda.bat`。在其他电脑使用前需修改该路径，或直接使用上述命令。

## Bubble Tracking：跟踪与运动分析

1. 选择气泡分割模型，在 **Input** 中选择视频或图像文件夹。
2. 设置 **Output Dir**，检查自动生成的 **Output Video** 和 **Output CSV** 路径。
3. 选择跟踪器、推理设备，设置输入帧率和空间标定。
4. 点击 **Start**，查看预览、进度、统计和日志；点击 **Stop** 可在帧间停止并保存已处理结果。

### 关键参数

以下为未加载本地保存状态时的界面默认值。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| Tracker | `botsort` | 可切换为 `bytetrack`；BoT-SORT 默认关闭 ReID |
| Confidence / NMS IoU | `0.10 / 0.45` | 检测置信度与非极大值抑制阈值 |
| Input Width / Height | `640 / 640` | 推理输入尺寸 |
| FPS | `30` | 图像序列的采样帧率；视频优先使用文件中有效的 FPS |
| Pixel Scale | `1.0` | 每像素对应的实际长度，单位由 Distance Unit 指定 |
| Distance Unit | `mm` | 长度单位标签；单独改变标签不会完成标定 |
| Speed Frame Gap | `1` | 计算速度时参考的历史观测间隔 |
| Min Track Frames | `5` | 轨迹达到指定观测次数后才进入有效运动统计 |
| Max Turn Angle | `120°` | 超过阈值的急转轨迹被判为无效 |
| Edge Margin (px) | `0` | 边缘排除距离；有掩膜时仍会排除触碰图像边界的实例 |
| Trail Length / Max History Gap | `24 / 10` | 轨迹显示长度与历史重置间隔 |
| Preview Stride | `5` | 预览更新间隔，不是输入抽帧间隔 |

### 标定与时间口径

若标定得到 **100 px = 1 mm**，应设置 `Pixel Scale = 0.01`、`Distance Unit = mm`。默认 `1.0` 只是初始值，不能直接作为真实实验标定。

```text
速度 = 两次观测之间的像素位移 × Pixel Scale / 时间间隔
时间间隔 = 处理序列中的帧索引差 / FPS
实际面积 = 像素面积 × Pixel Scale²
等效直径 = 2 × sqrt(实际面积 / π)
```

视频处理会读取有效的视频 FPS；若文件经过降速或变速导出，应核对该 FPS 是否仍代表实际采集时间。图像序列需手动填写实际采样帧率，抽帧后应使用抽帧后的等效帧率。当前流程不读取每张图片的独立采集时间戳。

图像坐标向右为 `+x`、向下为 `+y`，所以向上运动的 `vy` 为负值；`rise_speed = max(-vy, 0)`，向下运动时上升速度记为零。

界面 **Processing FPS** 和 **Frame Latency** 表示程序处理性能，不用于物理速度换算。跟踪页的计时主要覆盖推理及跟踪阶段，不包含完整的渲染、文件写入和界面刷新耗时。

### 输出与过滤

以输入文件夹 `test_images` 为例，Qt 界面通常生成：

```text
<Output Dir>/
├── test_images_result.mp4
├── test_images_motion.csv
└── test_images_result_tracks.txt
```

运动 CSV 每行对应一个通过过滤的轨迹样本，主要字段如下：

| 字段 | 含义与单位 |
| --- | --- |
| `frame`, `id` | 帧编号、跟踪 ID |
| `x`, `y` | 中心坐标，单位为像素 |
| `speed`, `avg_speed` | 当前速度、该轨迹已采样速度的均值，单位为实际长度/秒 |
| `area_mm2` | 实际面积；列名随 Distance Unit 变化 |
| `equivalent_diameter` | 等效直径，单位为实际长度 |
| `vx`, `vy`, `vx_avg`, `vy_avg` | 水平和竖直速度及其均值 |
| `dx_total`, `dy_total` | 相对有效轨迹起点的净位移 |
| `rise_displacement`, `rise_speed`, `rise_speed_avg`, `rise_total` | 上升方向的位移与速度统计 |
| `vertical_speed_abs`, `edge_margin_px` | 竖直速度绝对值、距图像边缘的像素距离 |

`rise_total` 是相对有效轨迹起点的向上净位移，不是累计上升路程。

预览中的标注和运动 CSV 受最小轨迹帧数、转角、历史长度和边缘条件影响。触发急转过滤时，该 ID 已积累的 CSV 行也会被移除；轨迹 TXT 则记录更早阶段的跟踪结果。因此 TXT、CSV、预览及界面统计的数量可能不同，运动 CSV 不能当作所有可见气泡的完整清单。

三张示例图片少于默认 `Min Track Frames = 5`，可能得到只有表头的运动 CSV。它们适合先检查图像读取和分割；定量速度分析应使用连续且时间间隔可靠的实验序列。

## Gas Holdup：ROI 面积占比分析

1. 打开 **Gas Holdup** 标签，选择气泡实例分割模型和 **Input Folder**。
2. 设置 **Output Overlay Folder** 与 **Output CSV**。覆盖图目录必须位于输入目录之外，避免再次被识别为输入。
3. 按需要设置置信度、最小面积、ROI、边缘策略和长度标定。
4. 点击 **Start**，检查每帧覆盖图、气泡数量、面积占比和平均等效直径。

该工作流目前通过界面处理图像文件夹；视频需先按所需间隔导出为图片。

### 计算口径

```text
Gas Holdup (%) = ROI 内有效实例掩膜的并集像素数 / ROI 像素数 × 100
```

- 面积来自实例分割掩膜，不使用检测框；掩膜重叠部分只计一次。
- `BubbleCount` 是通过边缘策略和最小面积筛选的实例数量。
- 最小面积默认 `10`，按实例在 ROI 内的像素面积筛选。
- ROI 使用 `X / Y / W / H`，默认覆盖整张图片；宽或高为 `0` 时延伸到对应图像边界。
- `BubbleAreaRatio` 和 `GasHoldup` 当前均为百分数，数值相同。例如 `25` 表示 `25%`，不是 `0.25`。
- 平均等效直径是各有效实例在 ROI 内的等效圆直径的算术平均；ROI 截断气泡时，直径对应截断后的面积。

这里的 Gas Holdup 是**二维图像中所选 ROI 的气泡面积占比**。将其解释为三维体积气含率或气体流量，需要额外的实验模型与验证。

### 边缘策略

| 策略 | 当前实现 |
| --- | --- |
| **Include** | 保留触碰图像边界的实例，计入 ROI 内可见掩膜面积 |
| **Exclude** | 排除触碰整张图像边界的实例；仅触碰 ROI 边界不会因此被排除 |
| **Weighted** | 计入 ROI 内可见掩膜面积；当前数值行为与 Include 相同，不估算图像外缺失面积 |

### 长度标定与分割质量

Gas Holdup 提供独立的 `Calibration Pixels`、`Actual Length` 和 `Distance Unit`。默认 **100 px = 1 mm**，对应 `0.01 mm/px`；请按实际实验修改。该页标定不会自动同步到跟踪页的 Pixel Scale。

```text
RealUnitsPerPixel = Actual Length / Calibration Pixels
AverageDiameter = AverageDiameterPx × RealUnitsPerPixel
```

标定只影响实际直径换算，不改变气含率百分比。可选高质量分割默认关闭；开启后使用更精细的掩膜处理，并可调整推理分辨率，但会增加显存和处理时间。

### 输出字段

每个已处理图像生成一张 `000001_<原图名>.png` 格式的覆盖图，包含灰度底图、实例掩膜、轮廓和统计信息；CSV 每图一行：

| 字段 | 含义 |
| --- | --- |
| `Frame`, `BubbleCount` | 顺序编号、有效气泡实例数 |
| `BubbleArea`, `ROIArea` | 掩膜并集与 ROI 的像素面积 |
| `BubbleAreaRatio`, `GasHoldup` | 面积占比，单位 `%` |
| `AverageDiameterPx`, `AverageDiameter` | 像素等效直径与标定后的等效直径均值 |
| `DiameterUnit`, `RealUnitsPerPixel` | 长度单位及换算比例 |
| `ProcessingTime` | 单图处理耗时，单位秒，包含读取、推理、分析、覆盖图生成与保存 |

默认输出通常位于输入文件夹的同级目录，命名为 `<文件夹名>_gas_holdup_overlays/` 和 `<文件夹名>_gas_holdup.csv`；也可手动指定。CSV 使用 UTF-8 BOM 编码，便于 Excel / WPS 读取。

## 项目结构

```text
HkcTracker/
├── main.py                      # 当前 Qt 桌面入口
├── bubble_tracker_ui_qt.py      # 主窗口、跟踪页控制与后台任务
├── bubble_tracker_ui.ui         # Qt Designer 界面文件，运行时需要
├── demo_yolo11.py               # 跟踪流程、运动指标、输出与默认参数
├── bubble_analyzer/
│   ├── inference.py            # 共用 YOLO 预测器工厂
│   ├── image_io.py             # 图像读取、自然排序、TIFF 处理
│   ├── gas_holdup.py           # ROI 和掩膜面积统计
│   ├── gas_holdup_workflow.py  # 批处理、覆盖图与 CSV 导出
│   └── gas_holdup_qt.py        # Gas Holdup 页面
├── detector_head/             # YOLO11 适配器及历史 YOLOv5 实现
├── tracking/                  # BoT-SORT 适配器与 ByteTrack
├── BoT-SORT/                  # 跟踪所依赖的第三方源码与配置
├── track_utils/               # 计时等辅助工具
├── visualization/             # 结果绘制；outputs/ 为本地产物
├── data/dataset/test_images/   # 小型测试图片
├── test/                      # 自动化测试及历史 Demo
├── weights/                   # 本地模型目录，权重不纳入 Git
├── ui_formatters.py           # 性能指标显示格式
├── requirements.txt           # 当前桌面流程的依赖
└── START.bat                   # 需适配本机 Conda 路径的启动脚本
```

`tracking/bot_sort.py` 直接引用 `BoT-SORT/`，发布时需保留该源码目录及第三方许可证。`bubble_tracker_ui.py` 是旧版界面，`main.py` 当前使用 Qt 版本。历史 YOLOv5 和 `test/demo/` 脚本保留供参考，部分包含原开发机的绝对路径。

## 测试与脚本运行

在项目根目录使用已安装依赖的环境执行：

```powershell
conda run --no-capture-output -n hkc python -m unittest discover -s test -p "test_*.py"
```

测试覆盖 ROI 与掩膜统计、边缘策略、长度标定、批处理输出、停止信号、共用推理配置、Qt 页面状态和性能指标显示。Qt 测试使用 offscreen 平台；测试通过不等于已验证真实模型精度或真实视频的跟踪效果。

不打开界面时，可运行已有默认跟踪脚本：

```powershell
conda run --no-capture-output -n hkc python demo_yolo11.py
```

该脚本使用文件中的 `Args` 默认配置，尚未提供 `--model`、`--source` 等命令行参数。运行前检查模型、输入、输出路径、FPS 和标定值；默认模型为 `3900.pt`，没有默认视频时使用测试图片目录。未指定显式输出路径时，结果保存到带时间戳的子目录。

## GitHub 仓库与本地文件

[.gitignore](.gitignore) 已覆盖以下本地文件与生成产物：

- 模型权重：`*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.engine`。
- 实验视频和演示视频：`data/videos/`、`BoT-SORT/assets/*.mov`。
- 分析结果：`visualization/outputs/`、Gas Holdup 默认命名的覆盖图目录和 CSV。
- 缓存、临时依赖与环境：`__pycache__/`、`.pytest_cache/`、`.deps/`、`.tmp/`、`.ultralytics/`、虚拟环境等。
- 本地界面状态 `.bubble_tracker_ui_state.json`、IDE 设置、训练输出和日志。

源码、`.ui` 界面文件、配置、测试、文档、第三方许可证和小型示例图片应保留。PNG/JPG 没有全局忽略，以便提交文档截图；大量实验图片和自定义输出目录请放在仓库外，或为其补充精确的忽略规则。

模型与大型数据可通过 GitHub Releases 或其他下载渠道单独提供，并记录模型用途、版本和获取方式。仅添加 `.gitignore` 不会停止跟踪已提交文件，也不会清除历史中的大文件；首次上传前可用以下命令检查仍被跟踪的忽略项：

```powershell
git ls-files -ci --exclude-standard
```

## 常见问题与测量限制

| 现象 | 检查项 |
| --- | --- |
| 找不到 `cv2`、`torch` 或 `PySide6` | 确认使用 `hkc` 环境，并在同一环境安装 `requirements.txt` |
| `lap` / `cython-bbox` 安装失败 | 检查 Python 版本及对应 wheel；需要本地编译时准备 C/C++ 构建工具 |
| 找不到模型或输入 | 克隆不包含权重和实验视频；重新选择本机路径 |
| Gas Holdup 拒绝模型 | 确认模型任务为实例分割，而不是普通目标检测 |
| 运动 CSV 只有表头或标注较少 | 检查序列长度、轨迹门槛、边缘、转角及置信度过滤 |
| 速度或直径量级不正确 | 核对采集帧率、Pixel Scale、Gas Holdup 独立标定和长度单位 |
| 覆盖图目录被拒绝 | 确保该目录不等于输入目录，也不是其子目录 |
| 找不到结果或仍显示旧路径 | 两个页面会保存本地状态；检查当前输出框中的路径 |

重复使用相同输出文件名会覆盖已有结果，建议为不同实验指定独立输出目录。

定量结果依赖模型分割质量、图像尺度、标定和采样时间。跟踪页缺少匹配分割轮廓时，中心与面积可能回退到检测框估计；Gas Holdup 则始终基于掩膜。两种流程的面积与样本选择口径不同，不能直接混作同一统计量。当前以分析与结果导出为主，尚未实现安全预警模块；性能与精度需结合具体模型、设备和实验数据评估。

## 致谢与许可

本项目基于或使用以下开源项目：

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [FastReID](https://github.com/JDAI-CV/fast-reid)
- [Qt for Python / PySide6](https://doc.qt.io/qtforpython-6/)

感谢原项目作者及 [海葵云](https://lyh.haikuicloud.com) 提供的项目基础和开发支持。原项目相关应用：[PyPeriShield](https://github.com/kendtank/PyPeriShield)。

第三方组件遵循各自许可证，详见 [BoT-SORT/LICENSE](BoT-SORT/LICENSE)、[FastReID LICENSE](BoT-SORT/fast_reid/LICENSE)、[YOLOv7 LICENSE](BoT-SORT/yolov7/LICENSE.md) 及各上游项目。当前仓库根目录尚未声明独立的项目许可证。
