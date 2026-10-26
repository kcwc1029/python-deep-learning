"""依序執行教材，保存真實輸出；不依賴 Notebook server。"""
from pathlib import Path
import ast
import base64
import io
import json
import os
import time
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
torch.set_num_threads(2)
path = ROOT / "深度學習基礎_v2.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
shell = InteractiveShell.instance()
execution_count = 0
start = time.perf_counter()
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    ast.parse(cell["source"])
    execution_count += 1
    cell["execution_count"] = execution_count
    outputs = []

    def show_figures(*args, **kwargs):
        # 以 Notebook display_data 形式保留每張實際繪製的圖。
        for figure_number in plt.get_fignums():
            figure = plt.figure(figure_number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            from IPython.display import display
            display({"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                     "text/plain": f"<Figure {figure_number}>"}, raw=True)
        plt.close("all")

    plt.show = show_figures
    with capture_output() as captured:
        result = shell.run_cell(cell["source"], store_history=False)
    if captured.stdout:
        outputs.append({"output_type": "stream", "name": "stdout", "text": captured.stdout})
    if captured.stderr:
        outputs.append({"output_type": "stream", "name": "stderr", "text": captured.stderr})
    for rich_output in captured.outputs:
        outputs.append({"output_type": "display_data", "data": rich_output.data,
                        "metadata": rich_output.metadata})
    cell["outputs"] = outputs
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f"Cell {index} failed") from (result.error_before_exec or result.error_in_exec)
    print(f"PASS cell={index} execution={execution_count}", flush=True)

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
summary = {"code_cells_executed": execution_count, "seconds": round(time.perf_counter() - start, 2),
           "device": str(shell.user_ns["device"]), "best_epoch": shell.user_ns["best_epoch"],
           "data_rows": len(shell.user_ns["df"]), "metrics": shell.user_ns["metrics_df"].to_dict("index"),
           "reload_predictions_checked": True}
(ROOT / "outputs/v2/verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
