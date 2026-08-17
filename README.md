# TF-IDF 双模块舆情分析

面向 iPhone 16 舆情的可复现 TF-IDF 基线仓库，分为两个互不混训的模块：

- `complaint`：黑猫投诉的产品主题、服务主题、严重度、解决状态、用户诉求。
- `ecommerce`：京东/淘宝电商评论的情感、产品主题、服务主题。只有京东人工标注参与训练，淘宝明确标为 `domain_transfer`。

项目不会提交 SQLite、人工标注 Excel、完整 CSV、用户名或模型二进制。模型、模型卡、完整评估报告和 SHA256 清单由 GitHub Release 提供。

## 安装

```powershell
git clone https://github.com/yxforever666gh/TF-IDF.git
cd TF-IDF
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

支持 Python 3.10–3.13；CI 固定验证 3.10 和 3.12。

## 本地数据布局

默认读取仓库同级的 `raw_data`，也可通过 `--data-root` 或环境变量 `TFIDF_DATA_ROOT` 指定。目录中应能唯一找到：

- `blackcat-complaint.sqlite3`（6,391 条投诉）
- `jingdong-comment.sqlite3`（1,271 条主评论、1,173 条回复）
- `taobao_comments.sqlite3`（2,400 条主评论、1,534 条回复）
- 文件名含 `1502条.xlsx` 的黑猫人工标注
- 文件名含 `1271条.xlsx` 的京东人工标注

源数据库以 SQLite 只读模式打开，准备前后会核对 SHA256。

## 命令

```powershell
tfidf-analytics complaint prepare
tfidf-analytics complaint train
tfidf-analytics complaint evaluate
tfidf-analytics complaint predict
tfidf-analytics complaint run-all

tfidf-analytics ecommerce prepare
tfidf-analytics ecommerce train
tfidf-analytics ecommerce evaluate
tfidf-analytics ecommerce predict
tfidf-analytics ecommerce run-all

tfidf-analytics verify
```

指定数据目录的示例：

```powershell
tfidf-analytics --data-root "D:\广州凯捷实习\数据清洗\raw_data" complaint run-all
```

对已有脱敏 CSV 预测（至少包含 `clean_text`）：

```powershell
tfidf-analytics ecommerce predict --input data/samples/ecommerce_sample.csv --output predictions.csv
```

## 固定输出

每个模块写入 `data/output/<domain>/`：

- `cleaned_records.csv`
- `all_predictions.csv`
- `analysis_ready.csv`
- `review_queue.csv`
- `training_report.json`
- `evaluation_report.json`
- `model_card.md`

`all_predictions` 保留所有候选模型结果。只有通过质量门槛的任务才会填充 `analysis_ready` 中对应的 `approved_*` 字段；其余任务标为 `baseline_only`。

电商评分严格保留 `original_rating`，仅从真实评分派生 `rating_sentiment`，绝不把模型情感转换成伪 1/3/5 星。回复保留在完整结果中，但不会进入默认评论 KPI。

## 训练与评估协议

- 按文本哈希和重复组确定性划分 70/15/15，随机种子 42，防止重复文本跨集合泄漏。
- 比较字符 TF-IDF、词级 TF-IDF、二者组合。
- 单标签比较类别加权 LinearSVC 与 SGD Logistic；多标签使用 One-vs-Rest SGD Logistic，并在验证集逐类调阈值。
- 仅用验证集选型，选定后在训练集+验证集重训，冻结测试集只评估一次。
- 投诉无法确认范围和明确旧型号进入复核队列；电商淘宝结果标记为迁移预测。

当前发布结果见 [reports/metrics-summary.json](reports/metrics-summary.json)。模型卡会披露逐类支持数、召回率、混淆矩阵和平台限制。

## 下载 Release 模型

安装 [GitHub CLI](https://cli.github.com/) 后：

```powershell
gh release download v0.1.0 --repo yxforever666gh/TF-IDF --dir release-v0.1.0
Get-FileHash release-v0.1.0\complaint-model-v0.1.0.zip -Algorithm SHA256
Get-Content release-v0.1.0\SHA256SUMS.txt
```

解压对应模型包后，可将 `model.joblib` 放到 `artifacts/complaint/` 或 `artifacts/ecommerce/`。只应加载可信来源且 SHA256 校验一致的 Joblib 文件。

## 隐私与测试

```powershell
python -m ruff check src tests
python -m pytest
```

公开合成样本位于 `data/samples/`。CI 不依赖私有原始数据，并运行静态检查、单元测试、合成训练/预测测试和样本隐私扫描。

## 许可证

[MIT](LICENSE)
