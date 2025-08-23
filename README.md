# KoCEM: A Multimodal Knowledge and Reasoning Benchmark for Korean Construction Engineering & Management

## Leaderboard

<p>Per-locale rankings aggregated across splits. Columns show subject accuracies. Higher is better.</p>

### Locale: en

<table>
<thead>
<tr><th>Rank</th><th>Model</th><th>Architectural_Planning</th><th>Building_System</th><th>Comprehensive_Understanding</th><th>Construction_Management</th><th>Domain_Reasoning</th><th>Drawing_Interpretation</th><th>Interior</th><th>Materials</th><th>Safety_Management</th><th>Standard_Nomenclature</th><th>Structural_Engineering</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>gpt-4.1</td><td>13.64%</td><td>12.96%</td><td>100.00%</td><td>0.00%</td><td>30.77%</td><td>66.67%</td><td>16.67%</td><td>0.00%</td><td>25.00%</td><td>0.00%</td><td>50.00%</td></tr>
</tbody>
</table>


## Evaluation Guidelines
This repo provides a simple CLI to run evaluation over KoCEM subjects, produce outputs, and update the README leaderboard.

### Output layout
Results are saved under `output/<prompt>/<locale>/<model>/<split>/<subject>/` as:

```
output/
    <prompt>/            # e.g., mcqa (preferred) or test (fallback)
        <locale>/          # e.g., en, ko
            <model>/         # e.g., gpt-4.1
                <split>/       # dev, test, val
                    <subject>/   # e.g., Architectural_Planning
                        output.json        # per-sample entries with predictions and judges
                        evaluation.json    # per-sample judge details
                        result.json        # aggregated metrics (acc, std_dev, num_example, ...)
```

### Run inference
Use the `run_each` command to generate predictions and metrics.

```pwsh
uv run python -m app run_each --model gpt-4.1 --locale en --subjects Architectural_Planning --splits dev --prompt test

# Multiple subjects
uv run python -m app run_each --model gpt-4.1 --locale en --subjects '["Architectural_Planning","Materials"]' --splits '["dev","val"]' --prompt test
```

### Evaluate difficulties (optional)
If a subject provides difficulty labels, compute difficulty-wise metrics from saved outputs:

```pwsh
uv run python -m app eval_difficulties --model gpt-4.1 --prompt test --locale en
```

### Update README leaderboard
Generate per-locale, per-subject leaderboards and inject them into this README:

```pwsh
uv run python -m app.update_readme
```

The updater prefers `output/mcqa`; if absent, it falls back to `output/test`.

### Prompts
Prompts live under `prompts/<locale>/...`. Pick a prompt by name with `--prompt` (e.g., `test`, `mcqa`).

### Configuration
Set the following environment variables (via `.env` or shell) as needed:

- `DS_PATH`: Hugging Face dataset path (e.g., `pikaybh/KoCEM`)
- `DS_CACHE_PATH`: HF cache directory
- `OUTPUT_PATH`: Root directory for outputs (defaults to `./output`)
- Provider credentials (e.g., OpenAI) according to your model choice