#!/usr/bin/env bash
# Pre-push safety check — run before every `git push` to the public repo.
#   bash scripts/check_before_push.sh
set -u
fail=0
note() { echo "  $1"; }

echo "pre-push safety check:"

if git ls-files | grep -qx '.env'; then
  note "FAIL: .env is tracked  (fix: git rm --cached .env)"; fail=1
else
  note "ok:   .env not tracked"
fi

if git grep -qI "ABSK" -- . 2>/dev/null; then
  note "FAIL: an AWS bearer token (ABSK...) is in a tracked file"; fail=1
else
  note "ok:   no AWS token in tracked files"
fi

if git ls-files | grep -qE 'predictor/data/.+\.(jsonl|json)$'; then
  note "FAIL: dataset files under predictor/data are tracked (should be gitignored)"; fail=1
else
  note "ok:   dataset not tracked"
fi

if git ls-files | grep -qE '\.(safetensors|bin|gguf|joblib|pt|ckpt)$'; then
  note "FAIL: model weights are tracked"; fail=1
else
  note "ok:   no model weights tracked"
fi

if [ "$fail" -eq 0 ]; then
  echo "PASS — safe to push"
else
  echo "BLOCKED — fix the above, then re-run"
fi
exit "$fail"
