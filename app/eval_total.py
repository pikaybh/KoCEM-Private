import os, statistics
from utils.data import load_json, save_json
from schemas.kocem import KoCEM, Subject


def evaluate_total(
    model: str,
    split: str,
    prompt: str = "mcqa",
    locale: str = "en",
    **kwargs
) -> dict:
    # Use the project-wide default folder name 'output' if env is not set
    root_dir = os.getenv("OUTPUT_PATH", "output")
    output_path = os.path.join(root_dir, prompt, locale, model, split)

    pred_correct = 0
    correct_list = []  # 정답 여부를 기록할 리스트
    # KoCEM 스키마를 기반으로 subject 폴더명 -> dimension 매핑 구성
    subject_to_dimension: dict[str, str] = {}
    for name, val in KoCEM.__dict__.items():
        if isinstance(val, Subject):
            subject_to_dimension[name] = val.dimension

    # dimension 별 정답 통계 수집
    dim_correct_lists: dict[str, list[int]] = {}

    for entry in os.listdir(output_path):
        subject_dir = os.path.join(output_path, entry)
        eval_file = os.path.join(subject_dir, "evaluation.json")
        # Skip non-subject entries (e.g., result_*.json files) and missing evaluations
        if not os.path.isdir(subject_dir) or not os.path.exists(eval_file):
            continue
        data = load_json(eval_file)
        dim = subject_to_dimension.get(entry)
        if dim and dim not in dim_correct_lists:
            dim_correct_lists[dim] = []
        for _, result in data.items():
            if result['judge'] == 'Correct':
                pred_correct += 1
                correct_list.append(1)
                if dim:
                    dim_correct_lists[dim].append(1)
            else:
                correct_list.append(0)
                if dim:
                    dim_correct_lists[dim].append(0)
    # dimension별 결과 집계
    by_dimension = {}
    for dim, lst in dim_correct_lists.items():
        by_dimension[dim] = {
            'acc': (sum(lst) / len(lst)) if len(lst) > 0 else 0,
            'std_dev': statistics.stdev(lst) if len(lst) > 1 else 0,
            'num_example': len(lst),
        }

    result = {
        'model': model,
        'split': split,
        'acc': pred_correct / len(correct_list) if len(correct_list) > 0 else 0,
        'std_dev': statistics.stdev(correct_list) if len(correct_list) > 1 else 0,
        'num_example': len(correct_list),
        'by_dimension': by_dimension,
    }
    save_json(os.path.join(output_path, f"result_{model}_{split}.json"), result)
    return result


__all__ = [
    "evaluate_total"
]