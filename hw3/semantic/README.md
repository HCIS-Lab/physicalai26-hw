# Task 1 — Semantic Grounding (REUSE) + SHACL Validation

> 此為 demo 版:`ground_execution.py` / `shapes.ttl` 均為
> **完整參考解**;學生發放版(含 STUDENT TODO)在 `hw3_template/semantic/`。

學生實作兩件事,把 FK/IK 的**執行過程**帶進語意空間並驗證它:

| 實作 | 檔案 | 產物 |
|---|---|---|
| ① Grounding(REUSE 標準本體) | `ground_execution.py` 的 2 個 TODO 函式 | `output/data.ttl` |
| ② SHACL(驗證 shapes) | `shapes.ttl` 的 3 個 TODO shapes(2 個數值門檻 + 1 個 SHACL-SPARQL 關節限位) | `output/validation.ttl` |

SHACL 除了驗自己的 `data.ttl`,也必須驗**助教提供的 30 筆執行紀錄資料集**
(`ta-faulty-execution.ttl`:IK/FK/JointState 三類、好壞混合、含門檻與
限位邊界值)。驗證結果會逐筆與**標準答案 `ta-answer-key.json`** 比對——
每筆的旗標(`ARM_OUT_OF_RANGE` / `NO_CONVERGENCE` / `FK_INACCURATE` /
`JOINT_LIMIT_VIOLATION`)必須與答案**完全一致**(不多報、不漏報)才算寫對。
S3 = 8×(有問題案例答對率) + 2×(乾淨案例答對率)。

評分(30 分)由 `ta-shapes-full.ttl`(TA 完整 SHACL:STRUCTURE 結構檢查
+ 問題偵測 + SHACL-SPARQL 關節限位交叉檢查)與 `score_semantic.py` 自動執行:

```bash
PYTHON=~/miniconda3/envs/pdm-hw3/bin/python bash semantic/run_task1.sh --student-id <id>
```

五步:toolchain → Jena(shacl CLI)→ grounding → SHACL 三份驗證報告
→ 評分。設計原則:**SHACL 直接驗證數值**(門檻:0.90 m reach /
0.02 m residual / 0.005 m pose error),宣告式 constraints 取代
手寫驗證程式。

**Tasks 2–4 迴路**:`fk.py`/`ik.py`/`fsm_task.py` 跑完分數後會自動呼叫
`semantic/diagnose.py`——用**學生自己的** grounding 函式與 `shapes.ttl`
對執行紀錄做 SHACL 診斷,逐筆印出問題旗標(fail-soft:Task 1 未完成或
無 Jena 時只印一行略過訊息,不影響分數)。
