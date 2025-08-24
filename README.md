# KoCEM: A Multimodal Knowledge and Reasoning Benchmark for Korean Construction Engineering & Management

## Leaderboard

<p>Per-locale rankings. First, per split tables; then an overall weighted table. Columns show subject accuracies. Higher is better.</p>

## Locale: en

### Locale: en — Split: dev

<table>
<thead>
<tr><th>Rank</th><th>Model</th><th>Total</th><th>Architectural_Planning</th><th>Building_System</th><th>Comprehensive_Understanding</th><th>Construction_Management</th><th>Domain_Reasoning</th><th>Drawing_Interpretation</th><th>Interior</th><th>Materials</th><th>Safety_Management</th><th>Standard_Nomenclature</th><th>Structural_Engineering</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>claude-opus-4-1</td><td>81.82%</td><td>66.67%</td><td>80.00%</td><td>100.00%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2</td><td>gpt-4.1</td><td>78.72%</td><td>66.67%</td><td>80.00%</td><td>66.67%</td><td>100.00%</td><td>33.33%</td><td>100.00%</td><td>83.33%</td><td>87.50%</td><td>75.00%</td><td>100.00%</td><td>0.00%</td></tr>
</tbody>
</table>

### Locale: en — Split: test

<table>
<thead>
<tr><th>Rank</th><th>Model</th><th>Total</th><th>Architectural_Planning</th><th>Building_System</th><th>Comprehensive_Understanding</th><th>Construction_Management</th><th>Domain_Reasoning</th><th>Drawing_Interpretation</th><th>Interior</th><th>Materials</th><th>Safety_Management</th><th>Standard_Nomenclature</th><th>Structural_Engineering</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>claude-opus-4-1</td><td>81.52%</td><td>79.83%</td><td>83.65%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2</td><td>gpt-4.1</td><td>72.74%</td><td>75.92%</td><td>78.75%</td><td>45.96%</td><td>77.05%</td><td>47.84%</td><td>48.36%</td><td>64.15%</td><td>84.77%</td><td>73.14%</td><td>99.56%</td><td>54.68%</td></tr>
</tbody>
</table>

### Locale: en — Split: val

<table>
<thead>
<tr><th>Rank</th><th>Model</th><th>Total</th><th>Architectural_Planning</th><th>Building_System</th><th>Comprehensive_Understanding</th><th>Construction_Management</th><th>Domain_Reasoning</th><th>Drawing_Interpretation</th><th>Interior</th><th>Materials</th><th>Safety_Management</th><th>Standard_Nomenclature</th><th>Structural_Engineering</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>claude-opus-4-1</td><td>82.22%</td><td>80.49%</td><td>83.67%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2</td><td>gpt-4.1</td><td>69.55%</td><td>80.49%</td><td>81.63%</td><td>45.22%</td><td>70.59%</td><td>40.00%</td><td>55.56%</td><td>82.61%</td><td>90.70%</td><td>85.37%</td><td>100.00%</td><td>58.82%</td></tr>
</tbody>
</table>

### Locale: en — All splits (weighted)

<table>
<thead>
<tr><th>Rank</th><th>Model</th><th>Total</th><th>Architectural_Planning</th><th>Building_System</th><th>Comprehensive_Understanding</th><th>Construction_Management</th><th>Domain_Reasoning</th><th>Drawing_Interpretation</th><th>Interior</th><th>Materials</th><th>Safety_Management</th><th>Standard_Nomenclature</th><th>Structural_Engineering</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>claude-opus-4-1</td><td>81.59%</td><td>79.80%</td><td>83.61%</td><td>100.00%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2</td><td>gpt-4.1</td><td>72.43%</td><td>76.24%</td><td>79.10%</td><td>45.79%</td><td>76.85%</td><td>47.39%</td><td>50.00%</td><td>66.50%</td><td>85.37%</td><td>74.35%</td><td>99.60%</td><td>54.76%</td></tr>
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
				<split>/       # dev, test, val, extra
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
Generate per-locale leaderboards and inject them into this README:

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
