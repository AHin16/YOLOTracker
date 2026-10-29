# 气含率测算升级实施清单

对应规划：`tasks/plan.md`。日期：2026-10-22。

## 本轮规划交付

- [x] 核对当前分析、批处理、Qt、CSV 和测试入口。
- [x] 确定球形、椭球/轮廓、实验标定三条实施路线。
- [x] 写明分阶段任务、依赖、产物和验收条件。

## 阶段 0：测量合同与验证基线

- [ ] P0.1 固定面积/体积/标定字段、单位、状态与历史 CSV 兼容规则。
- [ ] P0.1 确认通道几何、厚度、长度标定和全深度成像条件。
- [ ] P0.2 定义解析几何样本、真实实验索引与参考值格式。
- [ ] P0.2 记录参考测量的空间范围、时间窗口及不确定度。
- [ ] P0.2 保存面积基线输出并记录已有测试结果。

## 阶段 1：球形端到端版本

- [ ] P1.1 实现逐泡球体积、测量体积和跨 ROI 的体积积分。
- [ ] P1.1 保留 ROI 截断前的形状，区分完整估算、部分贡献与不可用。
- [ ] P1.1 验证解析圆、重叠实例、厚度关系、单位和异常配置。
- [ ] P1.2 接入批处理、覆盖图、CSV 与质量状态，保持面积列兼容。
- [ ] P1.3 接入 Qt 方法/厚度参数与新旧配置恢复。
- [ ] M1 通过定向测试及假预测器端到端回放，人工检查一次界面和导出。

## 阶段 2：椭球端到端版本

- [ ] P2.1 提取轴长、方向、拟合残差和形状适用性。
- [ ] P2.1 实现显式第三轴假设、椭球体积和 ROI 体积截取。
- [ ] P2.2 接入方法选择、覆盖图、逐帧及逐实例明细。
- [ ] P2.3 用解析圆/椭圆、轴比和姿态样本比较球形与椭球。
- [ ] 验证拟合失败、倾斜、截断与遮挡状态，保存模型适用性记录。

## 阶段 3：轮廓积分与质量比较

- [ ] P3.1 实现对称轴、重采样、切片积分及 ROI 内体积计算。
- [ ] P3.1 使用解析旋转体和不同步长验证收敛。
- [ ] P3.2 输出边界、遮挡嫌疑、非轴对称、模糊及拟合质量原因。
- [ ] P3.3 同一次推理输出三模型对照与新增分析耗时。
- [ ] M2 完成固定真实图像集复核，保留失败案例和部分结果状态。

## 阶段 4：实验标定

- [ ] P4.1 导入 CSV/JSON 参考数据并验证单位与测量范围。
- [ ] P4.1 按采集时间聚合到参考窗口，拒绝全局/局部尺度不匹配。
- [ ] P4.2 实现少量固定特征的线性/岭回归及基线比较。
- [ ] P4.3 按实验批次/工况划分训练、调参与最终测试，防止相邻帧泄漏。
- [ ] P4.3 保存 JSON 模型包、特征单位、版本、配置与训练范围。
- [ ] P4.4 接入模型加载、窗口级结果、超范围和配置不匹配状态。
- [ ] M3 验证模型加载复现、数据对齐与真实留出结果；参考数据缺失时注明等待实测验证。

## 阶段 5：独立实验验收与交付

- [ ] P5.1 确定独立重复工况和训练前验收阈值，考虑参考方法不确定度。
- [ ] P5.2 比较面积、球形、椭球、轮廓、标定的误差与有效输出覆盖率。
- [ ] P5.2 按浓度、泡径、变形、边界、遮挡和清晰度复核失败案例。
- [ ] P5.3 更新公式、配置、CSV、状态和适用范围文档。
- [ ] P5.3 完成 hkc 测试、Qt 恢复/关闭检查与真实离线回放。
- [ ] M4 依据独立验证决定默认方法；未达标方法保持实验状态。

## 原始 Bubble Analyzer 完成记录（历史）

以下保留原始功能清单的完成状态，不代表新体积模型已实现或本轮重跑了测试。

## Shared Foundations

- [x] Extract tracker-free YOLO predictor construction.
- [x] Extract shared image discovery and loading.
- [x] Add tested mask-area and ROI statistics.
- [x] Add tested Include, Exclude, and Weighted policies.

## Gas Holdup Workflow

- [x] Process an input image folder without tracker imports.
- [x] Save one segmentation overlay per frame.
- [x] Write the required per-frame CSV columns.
- [x] Emit preview/progress metrics and support stopping.

## Qt Shell

- [x] Preserve Bubble Tracking as the first tab.
- [x] Add the independent Gas Holdup page.
- [x] Persist Gas Holdup settings under a namespace.
- [x] Coordinate close behavior for both workers.

## Verification

- [x] New focused tests pass.
- [x] Existing test suite passes.
- [x] Changed Python modules compile.
- [x] Offscreen Qt smoke test shows both tabs.
