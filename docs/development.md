# 开发与维护

普通用户请看[使用指南](user-guide.md)。2.0 使用 React + TypeScript 前端与 FastAPI 后端；用户安装、运行、更新入口不变，浏览器不需要 Node.js。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| [`easy_gui.py`](../easy_gui.py) | 薄启动器：保留应用互斥与更新恢复检查，然后启动 Web 服务 |
| [`tools/run_gui.py`](../tools/run_gui.py) | 原有运行环境预检与全生命周期应用锁 |
| [`tools/gui_server/`](../tools/gui_server/) | FastAPI 路由、业务服务、任务管理、SSE、翻译、安全与静态资源 |
| [`frontend/`](../frontend/) | React、TypeScript、Vite 开发源码，不随 ZIP 更新分发 |
| [`tools/gui_web/`](../tools/gui_web/) | 提交到 Git 的前端构建产物，ZIP 更新直接使用，不在用户机器构建 |
| [`indextts_training/`](../indextts_training/) | 数据集、校对、ASR、LoRA 训练、验证、ABX、检查点导出与跨任务互斥 |
| [`indextts/`](../indextts/) | 上游推理代码与少量兼容修改 |

不再使用 Gradio 构建页面；已有环境中的 Gradio 依赖无需卸载。上游 CLI 与 TensorRT 后端不属于本次 GUI 改造范围。

## 开发环境

Python 使用项目原有虚拟环境。前端维护者额外需要 Node.js 24 和 pnpm 12；版本锁定在[前端依赖清单](../frontend/package.json)与[锁文件](../frontend/pnpm-lock.yaml)中。

```powershell
# 首次准备前端开发依赖
pnpm --dir frontend install --frozen-lockfile

# 开发后端：使用隔离的合成测试数据，不接触真实音色与输出
.\.venv\Scripts\python.exe -B -m tools.gui_server.fake.seed_demo --data-dir .runtime/webui-demo
.\.venv\Scripts\python.exe -B -m tools.gui_server.launcher --fake --data-dir .runtime/webui-demo --no-browser --host 127.0.0.1 --port 7868

# 另一个终端启动 Vite；代理转发 API、SSE、校对与导出下载
$env:INDEXTTS_DEV_BACKEND = 'http://127.0.0.1:7868'
$env:INDEXTTS_DEV_PORT = '5180'
pnpm --dir frontend dev
```

模拟模式使用正弦测试音和假训练/导出进度，**不能评价声音质量或 GPU 兼容性**。页面始终有演示横幅。`--data-dir` 仅允许与模拟引擎一起使用；生产默认数据路径不变。初始化工具要求空目录或已有有效演示标记，并拒绝真实数据目录。重复初始化保留已有演示编辑。

后端是单进程、单 worker；不要设置多 worker：模型缓存、任务和事件订阅在进程内。多个维护者并行开发时必须分配不同端口与数据目录，构建产物只由一位维护者统一生成。前端 HMR 调试遇到跨模块状态不一致时完整刷新页面；最终验收必须针对实际构建产物。

## 数据与任务协议

- 现有音色及 TXT、情感描述、项目元数据、训练与验证清单、适配器、ABX 会话格式保留，不要求迁移。
- 新增合成历史元数据使用 WAV 相邻的 JSON 文件；无元数据的旧 WAV 仍可列出和播放。
- 普通 API 位于 `/api/`，类型由 FastAPI OpenAPI 自动生成到[前端声明](../frontend/src/api/schema.d.ts)。改动 Pydantic 模型后运行 `pnpm --dir frontend gen:api`，不要手改生成文件。
- API 错误统一为 `{error:{code,message,detail,...extra}}`；输入校验是 400，不是默认的 422；繁忙/冲突是 409；上传过大是 413。
- 校对仍使用 `/proofread/api/*` 的会话令牌与同源检查；旧 `/proofread/` 页面地址跳转到项目工作台，不再保留 iframe/内嵌 HTML。导出保留 `/exports/{task_id}/{kind}?token=...` 能力令牌链接。
- 长任务返回 202。推理、对话、ABX 共用 GPU 作业槽，并与训练/导出共用原子活动锁；不能采用“检查是否忙，然后启动”的有竞态方式。
- 停止是协作式操作：在文本块、对话轮次、ABX 音频、训练步之间处理，不强杀正在运行的 CUDA 内核。停止、失败、成功有不同终态。
- SSE `/api/events?lang=zh_CN` 首条为完整 `snapshot`，之后发送 job/status/console/training/export/system/settings 与心跳。断线重连重新同步快照；事件不是持久任务队列。进程退出后未完成作业不自动恢复，ABX 已保存片段可继续生成。
- 控制台保留 1400 行，以稳定行号原地替换进度条；清空批次包含 `reset:true`。界面保留最近 50 个已结束作业。
- 训练与事后验证共用 [`indextts_training/schedule.py`](../indextts_training/schedule.py) 的检查点/验证判定（含提前结束的 `end_step` 与区间密集验证）；模拟训练器也使用它。事后验证损失是 `validation_posthoc` GPU 作业，结果写入适配器目录的 `validation_posthoc.json`（按检查点名与验证清单指纹存储，权重文件大小/修改时间变化即过期），并重写 `tensorboard/posthoc/` 运行；曲线接口追加 `validation_posthoc/loss` 序列。
- 训练曲线从 TensorBoard event 文件读取，每条曲线最多返回 2000 点。TensorBoard 只绑定 `127.0.0.1`，不由服务器自动打开浏览器；远程访问需自行安全转发。

### 前端状态与修改保护

请求统一经过 [`api/client.ts`](../frontend/src/api/client.ts)，服务端返回的消息直接展示，不再当成前端翻译键。实时状态统一经过 [`api/live.ts`](../frontend/src/api/live.ts)；忽略旧连接事件，HMR 销毁时关闭 EventSource，重连后使查询缓存失效。

项目文件夹/语言修改必须调用校对面板的 `prepareContextChange(commit)`，**先**处理保存/放弃/取消，**再**执行绑定修改。面板在整个网络提交过程中保持锁定；失败保留可恢复草稿，取消不能先写服务器再回滚。父组件通过 `onContextLockedChange` 禁用克隆/删除，卸载时清理状态；修改前取消旧查询，避免迟到的响应覆盖新绑定。相关回归覆盖真实组件与 HTTP 调用顺序。

## 翻译与样式

前端英文原文通过 `t('...')` 翻译，中文词典按领域拆分在 [`frontend/src/i18n/zh_CN/`](../frontend/src/i18n/zh_CN/)。参数与枚举使用稳定英文/代码值，不把显示译文写入项目数据。后端使用延迟 `Message`，在 HTTP/SSE 输出时按请求语言渲染；原始日志与异常详情不强行翻译。

共享 UI 位于 [`frontend/src/ui/`](../frontend/src/ui/)，样式变量位于 [`frontend/src/styles/tokens.css`](../frontend/src/styles/tokens.css)。最低宽度 1024px，侧栏宽度 240px，1280px 以下折叠为 64px；尊重 `prefers-reduced-motion`，支持明暗主题和键盘操作。耗时动作需有 pending/禁用原因/进度/错误/完成反馈，不以控制台作为唯一反馈渠道。

排版使用 `--content-max` 内的 12 栏文字网格（`--grid-columns`、`--grid-gutter`）。页面统一使用 `PageHeader`：所属导航分组由 `AppShell` 通过 `PageEyebrow` 提供，标题/导语占 1–8 栏，操作按钮占 9–12 栏。`Section` 默认 `layout="tile"`（浅灰圆角块，用于合成、校对、训练等工作区）；设置和说明型内容使用 `layout="aside"`（细线分隔，标题列 1–4 栏、内容 5–12 栏，1200px 以下上下堆叠）。新增页面的分栏也应落在 12 栏上，例如 8/4 或 7/5。

页面过渡由 `ui/Motion.tsx` 的 `RouteTransition` 负责：`scope` 改变时用 Web Animations API 以平缓的 ease 曲线（`--ease-page`）播放 400ms 淡入上移，不使用 `key`，因此不会重新挂载子组件或丢失草稿。`AppShell` 以页面路径为 scope（项目工作台整体视为一页），`ProjectLayout` 以标签页为 scope。减少动态效果或不支持该 API 时直接跳过。

## 安全边界

这是可信本机/局域网工具，不包含账户鉴权；**不要直接暴露到公网**。文件浏览器是服务端文件浏览器，不是浏览器客户端磁盘。媒体接口只允许业务白名单路径；上传有 200 MiB 内容上限，外层请求体也限流，临时上传启动时及下次上传时清理 24 小时前的文件。

写 API 拒绝跨源请求，SPA 使用 CSP，所有 API 禁止共享缓存。危险操作使用二次确认；项目删除不会顺带删除训练输出。麦克风、系统通知依赖浏览器安全上下文：局域网明文 HTTP 通常不可用，此时界面提供原因，仍可上传音频。提示音在浏览器播放且受自动播放策略限制，不在服务端扬声器播放。

## 自动化检查

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -m unittest discover -s tests -p 'test_gui*.py'
& $py -B -m unittest discover -s tests -p 'test_*_gui.py'
& $py -B -m unittest discover -s tests -p 'test_updat*.py'
& $py -B -m unittest discover -s tests -p test_runtime_lock.py
& $py -B -m unittest discover -s tests -p test_easy_gui_i18n.py
& $py -B -m unittest discover -s tests -p test_funasr_asr.py
& $py -B -m unittest discover -s tests -p 'test_*proofread.py'
& $py -B -m unittest discover -s tests -p 'test_*export*.py'
& $py -B -m unittest discover -s tests -p test_merge_lora_to_pth.py
& $py -B -m unittest discover -s tests -p test_text_chunking.py
& $py -B -m unittest discover -s tests -p test_abx.py
& $py -B -m unittest discover -s tests -p test_version.py

pnpm --dir frontend typecheck
pnpm --dir frontend lint
pnpm --dir frontend check:i18n
pnpm --dir frontend test
pnpm --dir frontend build
```

API 测试用临时根目录、模拟模型与真实 HTTP/作业流，不能写入真实语料。训练验证核心的 [`test_training_validation.py`](../tests/test_training_validation.py) 使用 pytest；可使用单独的维护者测试环境，不修改用户的运行锁文件。在已安装 pytest 的测试环境中执行 `python -B -m pytest -q tests/test_training_validation.py --basetemp .runtime/pytest-validation`，将临时产物留在已忽略目录中。旧 Gradio 回调的 AST 测试已迁移至真实 API，校对浏览器状态断言由 React/Vitest 覆盖。

上游 `test_v1.py`、`test_v2.py`、`regression_test.py`、`padding_test.py` 需要真实模型和 GPU，不属于离线模拟验收。GPU 验收项目见[验收清单](webui-acceptance.md)。

## 构建与发布

源代码和构建产物必须在同一阶段提交。完成构建后再次构建并确认 [`tools/gui_web/`](../tools/gui_web/) 无差异；不要提交开发服务器或依赖目录。Vite 同时生成 `THIRD_PARTY_NOTICES.md`，Inter 字体许可由前端 public 目录复制；两者必须随构建分发。

ZIP 更新器只分发[更新清单](../tools/update-manifest.json)中允许的路径。新增根目录 `frontend/` 是开发专用，旧更新器会跳过它；运行时只依赖 `tools/` 下后端与已构建前端。**不要扩展旧更新器不认识的根路径来分发运行文件。**

增删程序文件或改变构建文件哈希后，显式暂存本阶段文件，再重新生成清单：

```powershell
# 先检查 diff，并 git add 本阶段的明确文件路径；并行开发时不要 blanket git add -A
$inventory = Join-Path $env:TEMP 'indextts-managed-files.txt'
git ls-files | Set-Content -Encoding utf8 $inventory
.\.venv\Scripts\python.exe -B tools/build_update_manifest.py --files-from $inventory
.\.venv\Scripts\python.exe -B tools/update_healthcheck.py --root .
```

发布前还须使用 **1.1.0 的 `parse_manifest`** 校验新清单，检查必需启动文件、静态资源和环境输入完整。更新健康检查只做语法、导入、API 路由与 SPA 资源检查，不加载模型、不启动浏览器。

[版本号](../tools/version.py)与[更新日志](CHANGELOG.md)同步维护。`pyproject.toml` 的 2.5.0 是上游包版本，不是 GUI 版本；它与 `uv.lock`、安装/验证脚本等属于环境输入，本次改造不改动它们，不要求重新安装。

开发分支的阶段推送：

```powershell
git push origin feat/webui-2
git push github feat/webui-2
```

**推送 GitHub main 才是正式发布**。未经明确批准，不推 main、不合并、不打发布标签。已有用户数据路径与环境哈希保持兼容，并不等于模拟测试已经验证真实 CUDA/训练表现。
