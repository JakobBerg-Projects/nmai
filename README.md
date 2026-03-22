# NM i AI

Solutions for the Norwegian AI Championship (NM i AI).

## Challenges

### Object Detection

Two-stage pipeline for detecting and classifying products on store shelves.

1. **Detection** — YOLOv8 trained as a binary detector (`nc=1`, product or not) and exported to ONNX. This lets all annotations train a single class, giving more data per class.
2. **Classification** — ResNet50 fine-tuned as a feature extractor. Reference images for each of the 357 product categories are pre-embedded. Detected crops are classified by cosine similarity against these reference embeddings.

Splitting detection from classification lets rare categories (few shelf annotations but available reference photos) still be recognised reliably.

### Astar Island

Probabilistic terrain prediction for a Viking civilization game, played over multiple rounds with a limited query budget.

- Learn transition priors from completed rounds (cross-seed learning).
- Build features per cell: distance to nearest settlement, coastal/forest adjacency, local settlement density.
- Select viewports using an entropy-aware value map — prioritise high-uncertainty, settlement-dense areas.
- Cache observations to disk so priors improve across runs.

### AccountingAgentV2

AI agent that automates tasks in Tripletex (Norwegian accounting software) via the Claude API.

- **Classify** the task using keyword matching across 7 languages into 20+ task types.
- **Pre-fetch** reference data (departments, accounts, customers, suppliers, currencies, etc.) so the agent can use IDs directly without extra lookups.
- **Fast path** — simple tasks are solved in a single tool call, skipping the full agent loop.
- **High-level tools** bundle multi-step workflows (e.g. create employee, register payment) to minimise API write calls, since each write reduces the score.
