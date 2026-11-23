# IMDb 影评洞察助手 | IMDb Review Insight

一个从课程项目文本情感分析模块延伸而来的 RAG 问答原型。系统按电影检索影评，并按引用编号定位原文。可接入本地 Ollama 和 Qwen2.5-1.5B 生成带引用的回答。

## 快速试用

安装 Python 3.9 或更高版本。在 Windows 上双击 `Start-Demo.cmd`，或运行下方命令。演示使用随项目提供的虚构影评，无需下载 IMDb 语料或模型。

```powershell
python app.py --corpus data/demo_reviews.jsonl --open-browser
```

真实 IMDb 评测的原始影评和模型权重由使用者在本地准备。仓库提供导入脚本、评测题、引用编号和汇总结果。真实数据导入和 Ollama 运行命令见下文。

## 评测入口

- [检索诊断与预训练重排](证据诊断报告.md)
- [生成评测与引用复核](生成评测报告.md)
- [证据门控](证据判断实验.md)、[原句筛选](原句筛选实验.md)与[答案字段核验](后验核验实验.md)

## v7 答案字段核验

v7 在独立实验脚本中比较原句筛选与答案字段核验。新电影留出集的逐题结果和复现命令见 [后验核验实验](后验核验实验.md)。

## v4 证据判断实验

新增了先检查证据再回答的实验模式。模型先判断候选片段能否直接回答问题，再复制证据原文。程序核对摘录确实连续出现在指定片段里，通过后才把摘录交给生成模型。判断为证据不足或摘录无法核验时，界面提示当前评论库的证据不足。证据与问题是否真的相关由评测另行核对。

同一套 BM25 候选片段下，可以用下面的命令试用。需要本机 Ollama 已运行并有 `qwen2.5:1.5b`。正常双击启动仍可以查看检索摘录。

```powershell
python app.py --corpus data/imdb_local.jsonl --ollama-model qwen2.5:1.5b --evidence-gate --port 0 --open-browser
```

实验包含原 50 题、两部新电影的 8 道开发题，以及另外六部新电影的 24 道留出题。模型只收到问题和当时返回的片段。详细对照与复现命令见 `证据判断实验.md`。打包的实验记录保留状态、引用编号和复核标签。完整原文和生成文字可在本地重新运行时查看。

## 打开页面

完整解压后双击 `Start-IMDb.cmd`。脚本会读取项目 `data` 目录中的原始 IMDb 压缩包，或复用 Windows 用户目录中已有的 TensorFlow Datasets IMDb 缓存。首次打开会导入 12 部作品，然后启动浏览器。需要 Python 3.9 或更高版本。启动脚本会检查 Python 是否可运行，也能识别装在用户目录里的 Anaconda 或 Miniconda。项目的检索和评测使用 Python 标准库。

如果想先看看界面，双击 `Start-Demo.cmd` 可以用三部虚构电影直接试用。两种入口都使用本机浏览器的 `http://127.0.0.1:7860`。关闭启动窗口即可停止程序。

选一部电影后，请用英文提问。试着选择 **Northfork**，输入 `What did reviewers think of the cinematography?`。点击回答中的证据编号，页面会跳到对应影评段落。

## 切换三种检索方式

双击 `Start-Semantic.cmd`。它会使用本机可用的 Python，首次运行会把需要的依赖和约 91 MB 的 MiniLM 权重放进项目目录。已安装 PyTorch 和 NumPy 的 Anaconda 环境可以直接使用。运行时会把评论转成向量，再打开页面。准备完毕后，页面的「检索方式」可以切换 **BM25**、**MiniLM** 与 **混合检索**。默认选中 BM25，它在本轮评测里找回了最多已标注的相关影评。 [MiniLM 模型卡](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)

要复现检索对照，先启动一次语义版，完成模型和环境准备，再运行下面的评测命令。

```powershell
python compare_retrieval.py --out data/imdb_retrieval_comparison.json
```

## 检索对照结果

三种方式使用相同的 360 条影评和分段，按电影过滤，每种最多返回 5 条不同影评。MiniLM 使用归一化向量的余弦排序，混合方式使用固定参数的倒数排名融合。Recall@5、MRR@5 和命中题数基于有已标注证据的 40 道题。

| 方法 | Recall@5 | MRR@5 | 前五至少命中一条的题数 |
| --- | ---: | ---: | ---: |
| BM25 | 54.9% | 0.715 | 38 / 40 |
| MiniLM | 51.8% | 0.704 | 35 / 40 |
| 混合检索 | 52.3% | 0.759 | 36 / 40 |

MiniLM 在 9 道题上找回更多相关影评，在 13 道题上召回变少。例如，`imdb-11` 问《穆赫兰道》哪些地方像希区柯克电影，BM25 找回 1 条，MiniLM 找回了两条已标注影评。混合检索的 MRR@5 最高，表示它把首条相关影评排得更靠前。逐题排名和本机 CPU 的检索延迟见 `data/imdb_retrieval_comparison.json`。

## 导入原始 IMDb 影评

下载 `aclImdb_v1.tar.gz` 后，把文件放入 `data` 目录。也可以用命令指定已有文件的路径。项目利用压缩包内原始 URL 的电影 ID 分组，原始 IMDb 文本由本地导入。已核对 Paddle 镜像与原始压缩包一致，SHA-256 为 `c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`。 [Stanford 数据集](https://nlp.stanford.edu/~amaas/data/sentiment/index.html) · [Paddle 下载镜像](https://bj.bcebos.com/dataset/imdb%2FaclImdb_v1.tar.gz)

```powershell
python import_imdb.py data/aclImdb_v1.tar.gz data/imdb_local.jsonl --split test --max-movies 12 --titles data/imdb_titles.json
python app.py --corpus data/imdb_local.jsonl
```

## 启用本地大模型生成

本轮用 Ollama 的 `qwen2.5:1.5b` 跑了 50 道题，固定输入为 BM25 返回的前 5 个片段。模型输出结论和证据编号，程序按编号回填原文。界面的日常启动方式可以继续用于检索摘录。 [Qwen2.5 官方模型标签](https://ollama.com/library/qwen2.5:1.5b-instruct-q4_K_M) · [Ollama 结构化输出文档](https://docs.ollama.com/capabilities/structured-outputs)

```powershell
ollama pull qwen2.5:1.5b
python app.py --corpus data/imdb_local.jsonl --ollama-model qwen2.5:1.5b --open-browser
```

如果在单独的本地端口运行 Ollama，用 `--ollama-url` 指定地址。服务端会校验引用编号是否属于本次候选片段，逐条结论是否得到原文支持记录在评测报告里。

### 复现生成评测

```powershell
python evaluate_generation.py --model qwen2.5:1.5b --out data/my_generation.json
python summarize_generation_audit.py
```

生成评测可断点续跑。输出旁边的本地检查点保留实际送入模型的片段，方便逐题核对。生成复核汇总命令默认读取随包提供的固定报告和标注。复核结果、典型错例和下一阶段的判定标准见 `生成评测报告.md`。

## 复现实验

本次整理了 12 部作品的 360 条真实影评，以及 Codex 阅读原文后编写的 50 道题。40 道有证据，10 道的答案无法从入库评论确认。标签按影评 ID 标注，各题记录了精简的证据说明。小规模评测适合诊断错例，后续可对相关性判断做人工复核。

```powershell
python evaluate.py --corpus data/imdb_local.jsonl --questions data/imdb_eval.jsonl --out data/imdb_baseline.json
```

| BM25 摘录基线 | 结果 |
| --- | --- |
| Recall@5，40 道有证据题的已标注评论召回比例取平均 | 54.9% |
| MRR@5，第一条命中的平均倒数排名 | 0.715 |
| 有证据题返回摘录 | 40 / 40 |
| 无证据题正确拒答 | 0 / 10 |

无证据题包含与评论重合的词，比如片中地点或被讨论的拍摄手法。BM25 会召回提到这些词的片段，但是片段可能没有问题所需的事实。生成评测额外核对了这些片段能否支撑实际回答。

## 目录

| 文件 | 内容 |
| --- | --- |
| `app.py`、`static/index.html` | 本地服务和可点击证据的界面 |
| `review_rag.py`、`retrieval.py` | 分段、三种检索方式、摘录和 Ollama 生成接口 |
| `embedding.py`、`prepare_semantic.py` | MiniLM 编码、模型与本地依赖准备 |
| `import_imdb.py` | 从原始压缩包导入电影、影评和稳定 ID |
| `evaluate.py` | 计算评论级 Recall、MRR 和拒答表现 |
| `compare_retrieval.py`、`data/imdb_retrieval_comparison.json` | 三种检索方式的对照实验与逐题结果 |
| `data/imdb_eval.jsonl`、`data/imdb_baseline.json` | 真实影评评测题、证据标注和逐题结果 |
| `evaluate_generation.py`、`data/imdb_generation.json` | 真实 Ollama 模型的生成评测、逐题状态和引用编号 |
| `summarize_generation_audit.py`、`data/imdb_generation_audit.jsonl` | Top 5 证据覆盖、答案质量和逐条引用复核 |
| `compare_answer_verification.py`、`data/imdb_verify_results.json` | 同一批原句上核验答案字段，并记录逐题对照结果 |
