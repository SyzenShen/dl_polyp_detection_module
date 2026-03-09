**《深度学习》课程设计报告**

  -------------------------------------------------
      题 目：       结肠镜息肉检测与报告生成系统
  --------------- ---------------------------------
      姓 名：                  申逸卓

      学 号：               202241000228

      班 级：                GJ生科22-2

      专 业：                GJ生物科学

      学 院：                  生科院

    指导老师：    

    完成时间：    
  -------------------------------------------------

信息与智能科学技术学院

2026年1月

# 目录 {#目录 .TOC-Heading}

[1 前言 [1](#_Toc201476105)](#_Toc201476105)

> [1.1 实验目的 [1](#_Toc201476106)](#_Toc201476106)
>
> [1.2 运行环境 [1](#_Toc201476107)](#_Toc201476107)

[2 基本功能 [1](#_Toc201476108)](#_Toc201476108)

> [2.1 基本功能介绍 [1](#_Toc201476109)](#_Toc201476109)
>
> [2.2 基本功能模块图 [1](#_Toc201476110)](#_Toc201476110)

[3 详细设计 [1](#_Toc201476111)](#_Toc201476111)

> [3.1 模型构建与训练 [1](#_Toc201476112)](#_Toc201476112)
>
> [3.1.1 数据集准备 [1](#_Toc201476113)](#_Toc201476113)
>
> [3.1.2 模型构建 [1](#_Toc201476114)](#_Toc201476114)
>
> [3.1.3 模型训练 [1](#_Toc201476115)](#_Toc201476115)
>
> [3.1.4 模型评估 [1](#_Toc201476116)](#_Toc201476116)
>
> [3.2 主要功能流程图 [1](#_Toc201476117)](#_Toc201476117)

[4 系统实现与测试 [2](#_Toc201476118)](#_Toc201476118)

> [4.1 主要功能实现（含模型部署） [2](#_Toc201476119)](#_Toc201476119)
>
> [4.2 相关功能代码 [2](#_Toc201476120)](#_Toc201476120)
>
> [4.3 页面实现 [2](#_Toc201476121)](#_Toc201476121)
>
> [4.4 最终实现效果 [2](#_Toc201476122)](#_Toc201476122)

[5 课程设计总结 [2](#_Toc201476123)](#_Toc201476123)

[6 参考文献 [2](#_Toc201476124)](#_Toc201476124)

# 1 前言

## 1.1 实验目的

本次《深度学习》课程设计的目标是做一个"从模型到系统"的完整小项目，而不是只停留在模型训练阶段。具体来说，本项目希望完成：

\- 使用目标检测模型自动标出结肠镜图像中的息肉区域；

\- 在网页界面中展示检测结果和标注框；

\-
利用大模型（Phi‑3）风格的文本生成模块，对检测结果做成条理清晰的"医学工程报告"；

\- 把整个流程集成到已有的 Django + Vue
文件管理系统中，让功能可以直接在浏览器里操作。

通过这个项目，我希望练习以下能力：

\- 理解并使用现成的目标检测模型（YOLO 系列）；

\- 了解深度学习项目从数据准备、模型训练到部署的基本步骤；

\- 将模型推理服务和前端页面打通，做一个可以演示的完整 Web 系统；

\-
体会工程中常见的问题，例如接口设计、前后端联调、界面细节和稳定性问题。

## 1.2 运行环境

本项目主要运行环境如下（版本为本次实验实际使用的版本，便于以后复现）：

\- 操作系统：Ubuntu Linux 20.04（服务器环境）

\- Python：3.8

\- 深度学习相关：

\- PyTorch（通过 Ultralytics YOLO 包间接使用）

\- Ultralytics YOLO11（用于目标检测）

\- Web 后端：

\- Django 2.1

\- Django REST framework

\- Web 前端：

\- Node.js 18

\- Vue 3 + Vite

\- Pinia（状态管理）

\- 浏览器：Chrome

硬件方面，训练和推理阶段可以使用带 GPU
的服务器；但为了便于演示，系统也支持在 CPU
上做小规模推理（速度略慢，但功能一致）。

# 2 基本功能

## 2.1 基本功能介绍

围绕"结肠镜息肉检测"这个具体场景，本系统实现了以下几个主要功能：

1\. 图像上传与管理

用户可以在 Web
页面中选择本地结肠镜图片上传，图片会通过前端发送到后端文件管理模块，存储在服务器上，同时在系统中建立对应的文件记录。

2\. 息肉自动检测

上传成功后，用户点击"Upload & Detect"按钮，后端会调用 YOLO11
模型，自动检测图像中的息肉区域，返回每个框的坐标和置信度。

3\. 检测结果可视化

前端使用 \`\<canvas\>\`
在原图上绘制绿色边框，并标出置信度百分比；同时在左侧以标签（chip）的形式展示所有检测结果。

4\. AI 报告生成（Phi‑3 风格）

用户点击"Generate AI Report
(Phi‑3)"按钮后，后端会将检测到的置信度信息和个数等统计特征交给一个"伪
Phi‑3 模块"，生成一段结构化、非诊断性的"医学工程报告"，前端以 Markdown
格式美观展示。

5\. 多级置信度提示

为了让界面更直观，系统根据置信度高低把检测结果标签分为不同颜色（如高置信度为黄色，其余为灰色），便于医生快速浏览。

整体上，系统实现了"上传图像 → 自动检测 → 可视化 → 报告说明"的闭环。

## 2.2 基本功能模块图

整个系统可以按功能分成四个主要模块：

\- 图像上传与管理模块（文件管理）

\- 深度学习检测模块（YOLO11）

\- AI 报告生成模块（Phi‑3 风格）

\- Web 展示模块（Vue 前端）

用简单的模块图表示如下（文字描述版）：

\- 前端（Vue）

\- 上传组件

\- 检测结果显示组件

\- 报告展示组件

\- 后端（Django）

\- 文件上传接口 \`/api/files/upload/\`

\- 检测任务接口 \`/api/ml/trigger/\`

\- 报告生成接口 \`/api/ml/{task_id}/explain/\`

\- 模型推理服务（YOLO11）

\- 报告生成服务（Phi‑3 Mock）

数据流向为：

\> 浏览器前端 → Django 文件上传 → YOLO11 检测 → Django 返回检测结果 →
前端画框

\> 前端 → Django 报告接口 → Phi‑3 文本生成逻辑 → Django 返回报告 →
前端展示

# 3 详细设计

## 3.1 模型构建与训练

## 3.1.1 数据集准备

本项目面向"结肠镜息肉检测"这个任务，选择了公开的 Kvasir‑SEG
数据集。该数据集提供了结肠镜图像以及对应的息肉标注。为了更方便使用 YOLO
检测模型，我对数据做了以下处理：

1\. 原始数据结构

\- 图像文件：\`images/\` 目录，格式多为 JPG/PNG（共1001张）

\- 标注文件：官方提供的遮罩

2\. 转换为 YOLO 格式

\- 编写脚本，将掩码或边界信息转成 YOLO 所需的 \`bbox\`（中心点坐标 +
宽高、归一化到 \[0,1\]）；

\- 每张图片对应一个 \`.txt\` 文件，内容类似：

\`0 x_center y_center width height\`

其中类别统一设为 0（polyp）。

3\. 划分训练集 / 验证集 / 测试集

\- 按图片随机划分，大致比例为：训练集 70%，验证集 20%，测试集 10%；

\- 目录结构整理为：

\`\`\`text

datasets/kvasir_yolo/

├─ images/

│ ├─ train/

│ ├─ val/

│ └─ test/

└─ labels/

├─ train/

├─ val/

└─ test/

\`\`\`

4\. 配置文件

为 YOLO 写了一个
\`data.yaml\`，包含数据集路径、类别数和类别名称（只有一个类别：\`polyp\`）。

以上步骤保证了数据能够被 Ultralytics YOLO 工具正常读取和训练。

## 3.1.2 模型构建

在模型选择方面，本项目没有从零搭建网络，而是采用 Ultralytics YOLO11
这一成熟的目标检测模型。选择的原因有：

\- 已经封装好训练和推理流程，易于上手；

\- 在检测任务上效果较好，社区资料较多；

\- 提供了 Python API，方便和 Django 一起集成。

模型构建的主要步骤是：

1\. 选择一个合适大小的 YOLO11
变体（比如较小的版本，适合课程时间和算力限制）；

2\. 使用官方预训练权重作为初始模型；

3\. 在我准备的息肉数据集上进行微调训练，使模型更适应结肠镜场景。

在代码层面，模型通过类似如下方式加载（伪代码）：

\`\`\`python

from ultralytics import YOLO

model = YOLO(\"yolo11n.pt\") 加载预训练权重

model.train(data=\"data.yaml\", \...) 在息肉数据集上微调

\`\`\`

## 3.1.3 模型训练

训练部分主要包含以下方面：

1\. 训练参数设置

\- 输入图片大小：640×640

\- 批大小（batch size）：根据显存自动选择

\- 训练轮数（epochs）：30 轮

\- 优化器与学习率：AdamW、0.002（但是optimizer=auto）

2\. 训练过程监控

\- 通过 Ultralytics 自带的日志，观察损失下降情况；

\- 在验证集上监控 mAP、召回率等指标，以判断是否过拟合或欠拟合。

3\. 模型保存

\- 训练过程中，自动保存验证集表现最好的权重，一般为 \`best.pt\`；

\- 这个 \`best.pt\` 就是后续部署与推理使用的模型文件。

对于课程设计来说，我并没有追求极致指标，而是更注重整个流程跑通。在实验中，模型在验证集上能够较稳定地标出大部分息肉，满足课程要求。

## 3.1.4 模型评估

模型训练完成后，需要做一个简单的评估，以确保它具备基本可用性：

1\. 定量评估

\- 主要关注 mAP@0.5 和召回率；

\- 在课程时间有限的情况下，只要在验证集上有一个比较合理的检测效果即可。

2\. 定性评估

\-
随机挑选若干结肠镜图像，让模型预测，并观察检测框是否覆盖了明显的息肉区域；

\- 注意观察：是否有大面积漏检、是否存在大量明显的错误框。

3\. 可视化结果

\- 将带框结果保存为图片，后续在系统中作为展示效果使用。

在本项目后续的 Web 系统中，模型评估实际上通过"界面可视化 +
人眼观察"的方式完成，这种方式对学习者来说更直观。

## 3.2 主要功能流程图

从用户角度看，本系统的主要流程可以概括为：

1\. 用户在浏览器打开"Polyp Detection"页面；

2\. 选择一张结肠镜图片并点击"Upload & Detect"；

3\. 前端将图片发送到后端，后端保存文件并触发检测任务；

4\. YOLO11 模型对图片进行推理，返回检测框和置信度；

5\. 前端根据结果在图片上画出框，并在左侧显示标签；

6\. 用户点击"Generate AI Report (Phi‑3)"；

7\. 后端根据检测结果构造简单描述，交给"Phi‑3 模块"生成文本报告；

8\. 前端以 Markdown 格式显示报告内容。

用简化的流程图文字说明：

\> 选择图片 → 上传 → 后端保存文件 → YOLO11 推理 → 返回检测结果 →
前端画框

\> → 用户点击生成报告 → 后端生成报告文本 → 前端展示报告

# 4 系统实现与测试

## 4.1 主要功能实现（含模型部署）

模型部署方式

在部署环节，我没有单独搭建复杂的模型服务，而是将 YOLO11 推理直接集成到
Django 后端中：

\- 在 \`ml_interface\` 应用中，创建了专门的视图和任务模型；

\-
模型在服务器启动时或第一次调用时加载为全局单例，避免重复加载造成开销；

\- 推理通过 Python 函数调用完成，返回 JSON 格式结果。

API 设计

和前端交互的核心接口有两个：

1\. 触发检测：

\`POST /api/ml/trigger/\`

\- 输入：\`task_type=\"polyp_detect\"\`, \`file_id\`

\- 输出：任务信息及检测结果（包括 \`detections\` 数组）

2\. 生成 AI 报告：

\`POST /api/ml/{task_id}/explain/\`

\- 输入：任务 ID

\- 输出：\`explanation\`（一段 Markdown 文本）

这样前端只需要记住一个 \`task_id\`，就可以分别请求检测结果和解释报告。

## 4.2 相关功能代码

下面选取两个关键代码片段进行说明。

## 4.2.1 报告接口（后端视图）

文件位置：\`ml_interface/views.py\` 中的 \`explain\` 动作。

\`\`\`python

class MLTaskViewSet(viewsets.ModelViewSet):

\...

\@action(detail=True, methods=\[\'post\'\])

def explain(self, request, pk=None):

task = self.get_object()

if task.task_type != \'polyp_detect\':

return Response({\'detail\': \'Only polyp_detect tasks can be
explained.\'},

status=status.HTTP_400_BAD_REQUEST)

if task.status != \'done\':

return Response({\'detail\': \'Task must be completed before
explanation.\'},

status=status.HTTP_400_BAD_REQUEST)

detections = task.result.get(\'detections\', \[\])

file_name = task.file.original_filename if task.file else
\"unknown_image\"

try:

client = Phi3Client()

explanation = client.generate_explanation(detections, file_name)

task.result\[\'explanation\'\] = explanation

task.save()

return Response({\'explanation\': explanation})

except Exception as e:

return Response({\'detail\': str(e)},
status=status.HTTP_500_INTERNAL_SERVER_ERROR)

\`\`\`

这个接口做的事情比较简单：

\- 从任务中取出检测结果；

\- 调用 \`Phi3Client\` 生成报告；

\- 保存并返回报告文本。

## 4.2.2 "伪 Phi‑3" 文本生成逻辑

文件位置：\`ml_interface/llm/phi3.py\`

本来准备用Phi-3来作为语言模型实现，但是在实际测试中稳定性和性能可被简单的Agent替代。代码会根据置信度高低，用不同的语气生成报告：

\`\`\`python

class Phi3Client:

\...

def generate_explanation(self, detections, file_name):

count = len(detections)

if count == 0:

prompt_context = f\"No polyps were detected in the image
\'{file_name}\'.\"

else:

confs = \[f\"{d\[\'confidence\'\]:.2f}\" for d in detections\]

prompt_context = (

f\"Detected {count} polyps in the image \'{file_name}\'. \"

f\"Confidence scores are: {\', \'.join(confs)}.\"

)

try:

if getattr(settings, \'USE_MOCK_PHI3\', True):

return self.\_mock_response(prompt_context, detections)

\...

except Exception:

return self.\_mock_response(prompt_context, detections)

def \_mock_response(self, context, detections=None):

if not detections:

return (

\"自动化技术报告\\n\\n\"

f\"输入数据: {context}\\n\"

\"\...\"

)

scores = \[float(d.get(\'confidence\', 0)) for d in detections\]

max_score = max(scores) if scores else 0.0

avg_score = sum(scores) / len(scores) if scores else 0.0

\...

if max_score \> 0.90:

高置信度：提示"高度疑似"，建议临床复核

elif max_score \< 0.30:

低置信度：提示结果不显著，建议人工检查

else:

中等置信度：给出"中等可信度"的措辞

\`\`\`

这段逻辑虽然没有真正调用远程大模型，但通过简单的规则，让报告内容看起来"有点聪明"，同时严格避免给出诊断结论。

## 4.3 页面实现

前端页面主要在 \`PolypDetect.vue\` 中实现。关键 UI 设计包括：

1\. 左侧区域：

\- 图片上传表单；

\- Detections 区域，使用"标签（chip）"横向展示每个检测结果；

\- "Generate AI Report (Phi‑3)" 按钮；

\- AI 报告区域，使用 Markdown 渲染，让标题和加粗效果更清晰。

2\. 右侧区域：

\- Visualization 面板，固定宽度；

\- 原始结肠镜图片；

\- 覆盖在图片上的 \`\<canvas\>\`，用于绘制绿色检测框和文字。

在前端，为了保证生成报告后图片不"跳动"，我：

\- 把图片外层容器宽度设置为固定值；

\- 每次图片加载完成后，根据图片实际渲染尺寸计算缩放比，再画检测框；

\- 当报告刷新时重新绘制一次框，防止布局微调造成偏移。

## 4.4 最终实现效果

本系统最终完成的效果如下（在报告中可以插入实际截图）：

1\. 初始界面：

只显示上传区域和"Upload & Detect"按钮，右侧暂不显示图像区域。

2\. 上传并检测后：

\- 右侧出现结肠镜图像，绿色框标出检测到的息肉，框上标有置信度百分比；

\- 左侧 Detections 区域以标签形式展示每个检测结果；

\- 高置信度标签为黄色；

\- 其余标签为灰色。

3\. 生成 AI 报告后：

\- 左侧"Medical Engineering Report"区域显示一段结构化的中文报告；

\- 报告中会说明检测到的息肉个数、置信度范围，并用简单语言给出建议；

\- 图片尺寸保持不变，检测框仍然准确贴合在对应位置。

在 Word 或 PDF 最终版中，可以插入以下截图（此处用占位符表示）：

\- 图 1 系统首页与上传界面

\`\![系统首页\](./images/home.png)\`

\- 图 2 检测结果和标注框可视化

\`\![检测结果\](./images/detect.png)\`

\- 图 3 生成 AI 报告后的完整界面

\`\![AI 报告界面\](./images/report.png)\`

# 5 课程设计总结

这次课程设计从"一个想法"走到了"一个可以在线演示的小系统"，整个过程给我的体会主要有以下几点：

1\. 深度学习项目不仅只有模型。

一开始我也倾向于只关注模型结构和指标，但在实现过程中发现，数据准备、接口设计、前后端联调、性能和用户体验，其实占了很大精力。

2\. 尽量使用成熟工具。

直接用 Ultralytics
YOLO11，比自己搭一个检测网络要省很多时间，也更稳定。省下来的时间，可以放在系统集成和功能细节上，这对课程设计更有价值。

3\. 接口要设计清晰。

把检测和解释拆成两个接口（\`trigger\` 和
\`explain\`），让前端的逻辑更清楚：先拿到检测结果，再按需生成报告，便于后续扩展。

4\. 前端细节会影响整体观感。

比如图片大小变化导致的框错位、报告的 Markdown
渲染、标签颜色的设计、Visualization
何时显示等等，这些问题看起来"不算深度学习"，但对最终的演示效果非常关键。

5\. "伪智能"也能发挥作用。

虽然没直接连上真正的 Phi‑3
服务，但通过简单的规则和模板，也可以做出比较自然的报告内容，既能展示思路，又能保证安全可控。

总体来说，这次课程设计帮助我把课堂上学到的深度学习知识和 Web
开发结合起来，完成了一个相对完整的小型工程。后续如果有时间，可以考虑接入真实的大模型
API，并继续丰富报告内容。

# 6 参考文献

1\. Ultralytics YOLO 官方文档：\<https://docs.ultralytics.com/\>

2\. Kvasir‑SEG 数据集介绍：

Debes, C., et al. "Kvasir-SEG: A Segmented Polyp Dataset."

3\. Goodfellow, I., Bengio, Y., Courville, A. Deep Learning. MIT Press,
2016.

4\. Django 官方文档：\<https://docs.djangoproject.com/\>

5\. Vue.js 官方文档：\<https://vuejs.org/\>

![](media/image1.png){width="5.768055555555556in"
height="3.6770833333333335in"}6. Vite 官方文档：\<https://vitejs.dev/\>
