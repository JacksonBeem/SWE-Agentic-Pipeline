from dataclasses import dataclass

# =========================
# HARD-CODED CONFIG
# =========================

OPENROUTER_API_KEY =   

ARCHITECT_MODEL = "google/gemini-3-pro-preview"
DEVELOPER_MODEL = "anthropic/claude-sonnet-4.5"
SECURITY_MODEL  = "openai/gpt-4o-mini"
QA_MODEL        = "openai/gpt-5.1"
VERIFIER_MODEL  = "google/gemini-3-pro-preview"

# =========================
# WORKFLOW PROFILE
# =========================
# One of: "monolithic(developer model)", "agentic", "agentic_plus_verifier"
WORKFLOW_PROFILE = "agentic"

# =========================
# DATASET SELECTION
# =========================
# One of: "humaneval", "bigcodebench", "mbpp"
ACTIVE_DATASET = "humaneval"

# Relative to pipeline/ directory.
DATASET_PATHS = {
    "humaneval": "data/human_eval.jsonl",
    "bigcodebench": "data/bigcodebench_200.jsonl",
    "mbpp": "data/mbpp_sanitized_200.jsonl",
}

# =========================
# DATA CLASSES
# =========================

@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None

@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 1500

@dataclass(frozen=True)
class PipelineConfig:
    run_id: str
    traceable: bool = True
    trigger_policy: str = "disagreement"
    allow_single_repair: bool = True
    enable_security: bool = False
    enable_verifier: bool = True

@dataclass(frozen=True)
class AppConfig:
    openrouter: OpenRouterConfig
    architect_model: ModelConfig
    developer_model: ModelConfig
    security_model: ModelConfig
    qa_model: ModelConfig
    verifier_model: ModelConfig


@dataclass(frozen=True)
class WorkflowDefaults:
    profile: str
    mode: str
    enable_security: bool
    enable_verifier: bool
    predictions_path: str
    executable_predictions_path: str
    boolean_results_path: str

# =========================
# CONFIG LOADER
# =========================

def load_config(run_id: str) -> AppConfig:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is empty (hardcoded config).")

    return AppConfig(
        openrouter=OpenRouterConfig(
            api_key=OPENROUTER_API_KEY,
            http_referer="http://localhost",
            x_title="Agentic Pipeline Experiment",
        ),
        architect_model=ModelConfig(model=ARCHITECT_MODEL),
        developer_model=ModelConfig(model=DEVELOPER_MODEL),
        security_model=ModelConfig(model=SECURITY_MODEL),
        qa_model=ModelConfig(model=QA_MODEL),
        verifier_model=ModelConfig(model=VERIFIER_MODEL),
    )


def get_workflow_defaults() -> WorkflowDefaults:
    profile = (WORKFLOW_PROFILE or "").strip().lower()

    if profile == "monolithic":
        return WorkflowDefaults(
            profile=profile,
            mode="monolithic",
            enable_security=False,
            enable_verifier=False,
            predictions_path="pipeline/logs/predictions_mono.jsonl",
            executable_predictions_path="pipeline/logs/predictions_mono_executable.jsonl",
            boolean_results_path="pipeline/logs/humaneval_boolean_results_mono.jsonl",
        )

    if profile == "agentic_plus_verifier":
        return WorkflowDefaults(
            profile=profile,
            mode="pipeline",
            enable_security=False,
            enable_verifier=True,
            predictions_path="pipeline/logs/predictions_agentic_plus_verifier.jsonl",
            executable_predictions_path="pipeline/logs/predictions_agentic_plus_verifier_executable.jsonl",
            boolean_results_path="pipeline/logs/humaneval_boolean_results_agentic_plus_verifier.jsonl",
        )

    # Default "agentic" (no verifier)
    return WorkflowDefaults(
        profile="agentic",
        mode="pipeline",
        enable_security=False,
        enable_verifier=False,
        predictions_path="pipeline/logs/predictions_agentic_no_verifier.jsonl",
        executable_predictions_path="pipeline/logs/predictions_agentic_no_verifier_executable.jsonl",
        boolean_results_path="pipeline/logs/humaneval_boolean_results_agentic_no_verifier.jsonl",
    )


def get_default_paths_for_dataset(dataset_type: str) -> tuple[str, str, str]:
    """
    Returns (predictions_path, executable_predictions_path, boolean_results_path)
    for the active workflow profile and requested dataset family.
    """
    wf = get_workflow_defaults()
    ds = (dataset_type or "").strip().lower()
    if ds not in {"humaneval", "bigcodebench", "mbpp"}:
        ds = "humaneval"

    base = f"pipeline/logs/{ds}/{wf.profile}"
    return (
        f"{base}/predictions.jsonl",
        f"{base}/predictions_executable.jsonl",
        f"{base}/boolean_results.jsonl",
    )


def get_active_dataset_type() -> str:
    ds = (ACTIVE_DATASET or "").strip().lower()
    if ds not in DATASET_PATHS:
        raise ValueError(f"ACTIVE_DATASET must be one of: {', '.join(sorted(DATASET_PATHS))}")
    return ds


def get_active_dataset_path() -> str:
    ds = get_active_dataset_type()
    return f"pipeline/{DATASET_PATHS[ds]}"
