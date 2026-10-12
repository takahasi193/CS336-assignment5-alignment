# GSM8K Prompting 基线评测综合报告 (Prompting Baselines Summary)

本报告针对斯坦福 CS336 作业五第一部分 **Problem (prompting_baselines)**，对预训练基座模型 `allenai/OLMo-2-0425-1B` 在三种不同 Prompt 策略下的答题与格式遵循表现进行系统性的量化统计与定性分析。

---

## 1. 实验设置 (Experimental Setup)

* **模型 (Model)**: `allenai/OLMo-2-0425-1B`（未经过微调与对齐的 1B 规模 Base 语言模型）
* **评测数据集 (Dataset)**: GSM8K 测试集 (`data/gsm8k/test.jsonl`)，共 1,319 道小学数学文字应用题
* **提示词策略对比 (Prompt Strategies)**:
  1. **`question_only` (零样本直接提问)**:
     ```text
     {question} Please put your final answer within \boxed{}.
     ```
  2. **`zero_shot` (R1-Zero 显式思维链提示)**:
     ```text
     A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
     User: {question}
     Assistant: <think>
     ```
  3. **`few_shot` (3-shot 示例示范)**:
     在 `zero_shot` 模板的基础上，在实际题目之前先提供了 3 个完整的问答示例（包含分步推理的 `<think>` 和规范输出的 `<answer>`）。
* **生成超参数 (Sampling Hyperparameters)**:
  * `temperature`: 1.0
  * `max_tokens`: 512
  * `n`: 1
  * `seed`: 42
  * `stop`: `question_only` 不设终止符；`zero_shot` 与 `few_shot` 设置为 `["</answer>"]` 且 `include_stop_str_in_output = True`。
* **判卷函数 (Grading Functions)**:
  * `question_only`: `cs336_alignment.drgrpo_grader.question_only_reward_fn`（解析 `\boxed{}` 中的答案）
  * `zero_shot` / `few_shot`: `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`（严格解析 `</think> <answer>` 与 `</answer>` 标签）

---

## 2. 评测指标与分类统计 (Evaluation Metrics - Deliverable a)

按照讲义要求，将所有 1,319 条生成结果划分为三大互斥类别：
* **Type 1**: `format_reward == 1` 且 `answer_reward == 1`（格式正确且最终答案正确）
* **Type 2**: `format_reward == 1` 且 `answer_reward == 0`（格式遵循但最终答案错误）
* **Type 3**: `format_reward == 0` 且 `answer_reward == 0`（格式违规/未按规范输出）

### 汇总对比数据表

| 提示词策略 (Prompt) | Type 1 (正确答案) | Type 2 (格式对/答案错) | Type 3 (格式违规) | 答题准确率 (Accuracy) | 格式遵循率 (Format Compliance) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`question_only`** | 2 | 166 | 1,151 | **0.15%** | 12.74% |
| **`zero_shot (r1_zero)`** | 0 | 625 | 694 | **0.00%** | 47.38% |
| **`few_shot (3-shot CoT)`** | **221** | **1,034** | **64** | **16.75%** | **95.15%** |
| **总测试样本量** | **1,319** | **1,319** | **1,319** | - | - |

---

## 3. 抽样质检与误差归因 (Qualitative Analysis - Problem a)

### (1) Category 2（格式正确但答案错误）质检分析（抽查 10+ 例）
在 `zero_shot` 的 625 个样本和 `few_shot` 的 1,034 个样本中，模型虽然成功输出了闭合标签，但未通过打分器判定：
* **是否存在被 Parser 误判的“冤假错案”？**
  * **结论：极少存在误判**。绝大多数样本确实存在计算错误或推导断裂，打分器表现稳健。
* **主要错误诱因拆解**：
  1. **字面复读模板说明（Instruction Parroting）**：在 `zero_shot` 下，模型因缺乏示例，直接将 Prompt 中的规则说明当成回答输出（例如直接输出 `<think> User's reasoning process here </think> <answer> ... </answer>`）。
  2. **在 `<answer>` 标签内输出未完成的算式**：例如输出 `<answer> The cost of 3 dozen donuts is 3 * 12 * $68. </answer>`，没有进行最终的四则运算化简，导致无法提取纯数值。
  3. **严重的算术幻觉与中间逻辑跳跃**：模型在推导过程中凭空制造不相干的负数、混乱的加减法，虽有推理形式但计算逻辑崩溃。

### (2) Category 3（格式违规）质检分析（抽查 10+ 例）
* **是否存在实际做对但因格式不合规被划入 Type 3 的情况？**
  * **结论：存在明显的假阴性（False Negatives）**！
  * **核心原因：打分器对换行符的严格匹配限制**：
    课程提供的 `r1_zero_reward_fn` 严格通过字符串匹配检查 `"</think> <answer>"`（中间必须为**空格**）。然而，基座模型在生成完思考链后，非常自然地习惯按回车换行输出解答：
    > **典型真实案例（第 2 题）**：
    > ```text
    > ...So, a robe takes 2 + (2 / 2) = 3 bolts of fiber in total.</think>\n<answer> It takes 3 bolts of fiber in total.</answer>
    > ```
    > 该题标准答案为 3。模型推导完全正确、答案正确、结构层次分明，但仅因使用了 `\n` 代替空格，直接被判定为 `format_reward = 0`，归入 Type 3 零分！
* **其他格式违规典型模式**：
  1. **标签拼写微瑕**：生成了 `</ think>` 或 `</ answer>`（标签内部多了一个空格），导致正则未能捕获。
  2. **陷入虚构剧本续写（Multi-turn Hallucination）**：模型在生成完思考后，自发伪造对话（如连续输出 `User: ... Assistant: ...`），漏掉了最终的 `<answer>` 闭合。
  3. **直接放弃作答（早停现象）**：在 `question_only` 中大量出现（首个 token 即为 `<|endoftext|>`，生成空串）。

---

## 4. 模型行为特征刻画 (Model Behavior Characterization - Problem b)

### 1. 仅提供题目是否足够引导模型作答？
**完全不够**。
* `OLMo-2-0425-1B` 作为一个预训练基座模型，本质是基于海量网页文本训练的 Next-Token 概率分布补全器，尚未经历任何对话或指令对齐。
* 当直接输入题目并要求使用 `\boxed{}` 时，模型无法理解“自己是一个需要做题的答题助手”。高达 **87.26%** 的样本直接放弃回答（输出空串）或误以为是习题集排版而继续生成 `Question 2:`、`Question 3:`，有效答题准确率仅为 **0.15%**。

### 2. Zero-shot `r1_zero` 如何塑造行为？
* 通过显式指定对话身份与前缀诱导（`User: ... \n Assistant: <think>`），**成功激活了模型的“解题角色”**。
* 格式遵循率从 12.74% 提升至 **47.38%**，有效遏制了交白卷的现象。
* 但由于没有提供上下文范例，1B 规模的小模型对复杂标签语法的自律性依然较差，且思维链推理缺乏约束，零样本下的最终准确率为 **0.00%**。

### 3. Few-shot `r1_zero_three_shot` 如何塑造行为？
* 3 个示范案例（In-Context Learning）带来了**质的飞跃**：
  1. **排版格式规范化**：格式违规率断崖式下降至 **4.85%**（格式遵循率高达 **95.15%**），模型完全掌握了标签的使用边界。
  2. **推理链路结构化**：示例向模型直观展示了“在 `<think>` 中用 3~4 步完成数值推导，在 `<answer>` 中仅保留纯数字”的标准模式。
  3. **答题准确率跃升**：最终准确率直接攀升至 **16.75%（成功答对 221 道题）**。

---

## 5. 对后续强化学习（GRPO）训练的指导意义

1. **基座模型蕴含推理潜力**：`few_shot` 取得的 16.75% 准确率证明，1B 基座模型底层已经掌握了解答小学数学应用题所必需的世界知识与代数逻辑，只是缺乏激活机制。
2. **GRPO 的核心任务**：
   - 传统提示工程（Few-shot）需要消耗大量 Prompt 上下文（增加推理成本与显存占用）。
   - 强化学习（GRPO）的目标正是：**在 Zero-shot 的简洁设定下，通过奖励信号与优势反传，自主激发模型内部沉睡的思维链推导能力，使模型在零样本下的准确率突破 25% 以上**。
