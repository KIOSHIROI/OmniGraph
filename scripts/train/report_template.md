# OmniGraph GVL Paper Report Template

## 1. Experiment Setup
- Date:
- Commit:
- GPU:
- Precision:
- Pipeline: `Stage1 -> Stage2A -> Stage2B -> Stage3 -> GQA val_balanced`
- Data:
  - Train: VG region + synthetic graph QA
  - Eval: GQA `val_balanced` (no GQA training labels)

## 2. Node Encoder Ablation (Fixed Recipe)
| Variant | Strict (all GT) | Coverage | Query (strict) | Notes |
| --- | ---: | ---: | ---: | --- |
| legacy_vg |  |  |  |  |
| open_vocab |  |  |  |  |
| hybrid |  |  |  |  |

## 3. Main Result
- Baseline strict: `0.3912`
- Current strict:
- Current query:
- Gate check:
  - strict target `>=0.4200`: PASS / FAIL
  - query target `>=0.2600`: PASS / FAIL
  - coverage target `>=0.999`: PASS / FAIL

## 4. Structural Breakdown (strict)
| Type | Accuracy |
| --- | ---: |
| choose |  |
| compare |  |
| logical |  |
| query |  |
| verify |  |

## 5. Error Analysis (Top 20)
- Query count failures:
- Object attribute confusion:
- Relation confusion:
- Multi-hop reasoning failures:

## 6. Training Stability
- OOM observed: YES / NO
- Peak VRAM:
- Early stopping trigger step:
- Best checkpoint path:
- Final exported state dict path:

## 7. Final Recommendation
- Selected model:
- Reason for selection (strict + query tradeoff):
- Next iteration plan (one paragraph):
