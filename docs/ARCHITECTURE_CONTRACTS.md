# Shared Contracts & Architecture Note

## 1. Overview and Core Principles

This document specifies the minimal shared contracts, schemas, configuration resolution mechanism, and local feedback persistence for the **AI ChatBot with Continuous Learning Pipeline**.

### Key Architectural Safeguards
1. **Separation of Concerns:** BERT classifies user intents; it does NOT replace Google ADK / Gemini. Gemini generates the conversational reasoning and response.
2. **Negative Feedback is NOT a Training Label:** A negative rating signals user dissatisfaction, not ground truth. Only feedback with an explicitly verified `corrected_intent` is flagged `training_eligibility = True` for supervised intent retraining. Uncorrected negative feedback is routed exclusively to drift analysis and prompt quality evaluation.
3. **No Retraining of Foundation Model:** Fine-tuning BERT updates the local intent classification weights. The Google ADK foundation model is never retrained; its behavior is guided through versioned, evaluated prompt instructions.
4. **Immutability of Known-Good Baselines:** Baseline instructions (`ChatBot/ADKBot/instruction.txt` as `v1.0.0`) and baseline model configurations are never overwritten. Candidate prompt and model configurations are registered with status `"candidate"` and evaluated before promotion.
5. **No External Database Overhead:** Minimal, robust, append-only JSON Lines (`user_feedback.jsonl`) persistence ensures thread-safe local storage for the deadline demonstration.

---

## 2. Shared Data Contracts (`common.contracts`)

All schemas are implemented using **Pydantic v2** (`BaseModel`) with strict type validation.

### A. Chat Request (`ChatRequest`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `message` | `str` | Yes | Non-empty user query (stripped, whitespace-only rejected). |
| `conversation_id`| `Optional[str]` | No | Session identifier (defaults to `None` / `"default-session"`). |
| `request_id` | `str` | No (Auto) | Unique correlation ID; auto-generated via UUID4 if omitted. |

### B. Chat Response (`ChatResponse`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `request_id` | `str` | Yes | Correlation ID matching the incoming chat request. |
| `response` | `str` | Yes | AI ChatBot Agent response text (also aliased as `.reply`). |
| `predicted_intent`| `Optional[str]` | No | Classified intent (e.g. `Greeting`, `TimeQuery`) when available. |
| `intent_confidence`| `Optional[float]`| No | Confidence score $[0.0, 1.0]$. |
| `model_version` | `str` | Yes | Active intent classifier version (e.g. `v0.1.0-baseline`). |
| `prompt_version` | `str` | Yes | Active ADK system instruction version (e.g. `v1.0.0`). |
| `latency_ms` | `float` | Yes | Total turn latency in milliseconds ($\ge 0.0$). |

### C. Feedback Request (`FeedbackRequest`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `request_id` | `str` | Yes | Target chat turn correlation ID. |
| `rating` | `FeedbackRating`| Yes | Normalized sentiment: `"positive"` or `"negative"` (accepts `+1`, `-1`, `thumbs_up`, `thumbs_down`). |
| `corrected_intent`| `Optional[str]`| No | Explicit ground-truth intent class provided by user/supervisor. |
| `corrected_response`| `Optional[str]`| No | Optional human-curated ideal response. |
| `textual_feedback`| `Optional[str]`| No | Qualitative user comment. |

### D. Feedback Record (`FeedbackRecord`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `record_id` | `str` | Immutable record ID (e.g. `fbr_9dcc540cf880`). |
| `request_id` | `str` | Correlated chat turn ID. |
| `message` | `str` | Original user query. |
| `predicted_intent` | `Optional[str]` | Predicted intent from inference time. |
| `rating` | `FeedbackRating` | Positive or Negative. |
| `corrected_intent` | `Optional[str]` | Ground-truth intent class (if verified). |
| `timestamp` | `str` | ISO-8601 UTC timestamp. |
| `model_version` | `str` | Classifier version active during turn. |
| `prompt_version` | `str` | Prompt version active during turn. |
| `training_eligibility`| `bool` | **True ONLY if `corrected_intent` is populated; False otherwise.** |
| `eligibility_reason` | `str` | Detailed audit reason for eligibility status. |

### E. Model Metadata (`ModelMetadata`)
Tracks BERT classification checkpoints across their lifecycle:
- `model_version`: Semantic version string (e.g. `v0.1.0-baseline`, `candidate-20261027`).
- `artifact_location`: Directory containing model weights and tokenizer.
- `dvc_revision`: Git or DVC commit hash representing dataset snapshot.
- `dataset_version`: Source dataset path (e.g. `ChatBot-PipeLine/data/raw/Intent.json`).
- `num_classes`: Number of target intents (`22`).
- `evaluation_metrics`: Dictionary (`macro_f1`, `accuracy`, `latency_p95_ms`).
- `promotion_status`: Lifecycle state (`"candidate"`, `"production"`, `"archived"`, `"rejected"`).

### F. Prompt Metadata (`PromptMetadata`)
Tracks Google ADK system instruction versions:
- `prompt_version`: Version string (e.g. `v1.0.0`, `v1.1.0-candidate`).
- `instruction_path`: Path to instruction text file.
- `associated_adk_model`: Associated LLM (e.g. `gemini-3-flash-preview`).
- `evaluation_status`: Lifecycle state (`"baseline"`, `"candidate"`, `"production"`, `"rejected"`).
- `evaluation_metrics`: Benchmark scores (e.g. `adherence_score`, `safety_score`).

---

## 3. Configuration Resolution Mechanism

The configuration resolver (`common.config_resolver`) enforces deterministic, single-point resolution:

```
[Runtime Override Parameter]
       │ (if absent)
       ▼
[Environment Variable (e.g. ACTIVE_PROMPT_VERSION)]
       │ (if absent)
       ▼
[ChatBot-PipeLine/configs/config.yaml]
       │ (if absent or invalid)
       ▼
[Known-Good Baseline Fallback (v1.0.0 / instruction.txt)]
```

- **`resolve_active_prompt_config()`**: Returns an immutable `ResolvedPromptConfig(prompt_version, instruction_text, instruction_path, metadata)`.
- **`resolve_active_model_config()`**: Returns an immutable `ResolvedModelConfig(model_version, artifact_location, metadata)`.

---

## 4. Local Persistence (`common.feedback_store`)

Feedback records are streamed to `ChatBot-PipeLine/data/feedback/user_feedback.jsonl`:
- **Append-Only:** $O(1)$ disk append with process-level locking.
- **Resilient:** Corrupt or partial lines are safely skipped during batch read without breaking the dataset.
- **Filterable:**
  - `store.get_training_eligible_records()`: Returns only records where `training_eligibility == True` for supervised retraining.
  - `store.get_drift_monitoring_records()`: Returns uncorrected and negative-only records for Evidently drift analysis.

---

## 5. Division of Work Across Developer Branches

| Developer | Branch | Primary Directory | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Dev 1 (ML Pipeline)** | `feature/ml-pipeline` | `ChatBot-PipeLine/` | `clean_data.py`, `tokenizer.py`, `model.py`, `train.py`, `predict.py`, `evaluate.py`, `dvc.yaml`, `retraining_pipeline.py`. Uses `from common.contracts import ModelMetadata, FeedbackRecord`. |
| **Dev 2 (ADK & Deployment)** | `feature/adk-deployment`| `ChatBot/`, Root | `ChatBot/ADKBot/api.py`, `agent.py`, Prometheus metrics, demo UI, `Dockerfile`, `.github/workflows/ci.yml`. Uses `from common.contracts import ChatRequest, ChatResponse, FeedbackRequest`. |

