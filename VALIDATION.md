# Verification performed on 1 October 2026

Local environment: Windows, Python 3.14.7. Exact dependencies are recorded in `requirements-lock.txt`.

- 30 automated tests passed. These include the complete offline pipeline, explicit failed-run status, deterministic robustness outputs, costs, causal transformations, cleaning, execution-price alignment, connected plateaus, skipped training guards and future/holdout isolation. The local invocation used `--basetemp runs/pytest_verify_20261001_1235` because the machine's existing system pytest temporary directory was inaccessible; no system permissions were changed.
- Ruff checks and formatting passed for source, tests and the parity verification utility. `pip check` reported no broken requirements.
- The five saved earlier fixed-parameter baselines were replayed without a network request. All raw factors matched within 1e-8, all positions matched exactly, and the maximum per-bar PnL difference was 1.1102230246251565e-16. The `02_03` check verifies parity with the previous reconstruction, not its original HTML, which had 412 position mismatches.
- A real `OPT25D_OI_03_01` run completed all 1,183 training candidates, 20 development test folds, a separate three-month holdout, cost/slippage/delay/neighborhood checks, 500 bootstrap draws and 100 conditional placebo draws. Warm-up observations were downloaded successfully from Cybotrade; existing cached research observations were reused. The final development test stopped before the holdout execution-price boundary.
- The real run selected `model=ezscorev1`, `window=120`, `threshold_1=0.6`, `threshold_2=-0.8` on the final pre-holdout training window. Its walk-forward Sharpe was approximately 0.6295 and holdout Sharpe approximately 0.8731. These scores differ from the earlier script because the approved pipeline uses extra warm-up, training eligibility, connected plateaus, execution-time purging and an untouched holdout. They are current-snapshot research results, not a claim of future performance.
- The generated HTML was inspected in a browser: all four interactive charts, metric/fold tables and methodology sections rendered, with no browser errors or document-level horizontal overflow.
- The source staged for Git was checked against the locally configured API key. No credential, downloaded observations, run outputs or environment files were included.

GitHub CI is configured for Windows/Linux on Python 3.13/3.14. Those remote jobs are distinct from the local checks above and are not claimed as completed here.
