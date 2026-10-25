# 現在のモデル・追加学習コマンド

2026年10月25日。日本語75.06%の採用GRUは学習済みです。利用は `Start_Kotoba.bat` をダブルクリックします。再学習は不要です。詳細な結果・制約は [RESULTS.md](../reports/RESULTS.md) に記録しています。

## このPCで追加実験する場合

以下は現在のローカル音声・特徴・固定インデックスを使い、別の実験フォルダへ保存するPowerShellコマンドです。既定モデルは自動的には置き換えません。

```powershell
cd C:\Users\harus\Downloads\Lipsync
.\.venv\Scripts\python.exe scripts\run_gpu_training.py `
  --architecture gru --channels 96 --gru-layers 2 `
  --skip-lags 0,2,4,6,8,10,12,14 --skip-hidden 32 --lookahead 10 `
  --index cache/features_v2/index_extended_complete_v3.json `
  --corpus cache/temporal_corpus/joint_complete_v3 `
  --breath-manifest data/breath_cc0/manifest_v2.json `
  --run my_gru96_experiment --epochs 30 --steps-per-epoch 300 `
  --batch-size 64 --english-weight 1.5 --threads 2 --device cuda `
  --gain-augment-db 8 --gain-augment-events-only --seed 42
```

上記は新しい重みから学習します。採用モデルは複数段階の追加学習を経たものなので、上記1回で同じ重み・75%以上を再現すると保証するものではありません。採用時の設定・履歴の要約は [実行記録](../reports/experiments/2026-10-25_final_training_run.json) に保存しています。

このPCの保存済みPyTorch重みから継続したい場合は、上記へ次を追加します。特徴の正規化も引き継ぎ、最適化器は新しく開始します。

```powershell
  --resume cache/tcn_experiments/gru96_complete_v3/best.pt
```

`cache/torch_cuda` にCUDA版PyTorchがあればランチャーが優先します。現在の環境はPyTorch 2.14.1+cu130、RTX 5060です。CUDAがない場合はCPU版PyTorchを入れ、`--device cpu` へ変更できます。CPUは所要時間が増えます。推論用アプリにはどちらも必要ありません。

```powershell
# CPU用の学習依存を準備する場合
.\.venv\Scripts\python.exe -m pip install -e ".[train,temporal-train,test]"
```

途中で止める場合、別のPowerShellで次の空ファイルを作ると現在のエポック終了後に停止し、最良・最終重みを保持します。同じ実験名で再開する際は、自分で作成したSTOPファイルを外し、`--resume` で保存重みを指定します。

```powershell
New-Item -ItemType File -Path cache/tcn_experiments/my_gru96_experiment/STOP -Force
```

## 評価の順序

学習結果は `cache/tcn_experiments/my_gru96_experiment/` の `best_int8.npz`、`best.pt`、`last.pt`、`history.json` に出ます。口形の確率補正と呼吸のしきい値は開発データだけで選択します。採用時は100 ms先読みの本体に20 msの確率補正を追加しました。

第1回の未使用セットは開発用へ移動済み、第2回セットは最終評価で使用済みです。これらを再評価して新しい未使用テストとは呼ばないでください。第3回候補の `cache/features_fresh_v4/index.json` は準備のみで、候補の推論は未実施です。呼吸の第2回テストは使用済みです。

採用にはモデルのハッシュと凍結した評価を照合する `scripts/promote_candidate.py` を用いました。採用時の具体的なコマンドは次です。単に学習しただけの新候補には使わず、別の未使用データで評価してからにしてください。

```powershell
.\.venv\Scripts\python.exe -m scripts.promote_candidate `
  models/candidates/kotoba_gru_candidate_v3.npz `
  reports/experiments/2026-10-25_fresh_attempt_2.json --minimum-japanese 0.75
```

## 推論速度・動作の確認

```powershell
.\.venv\Scripts\python.exe -m kotoba_lipsync devices
.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile monitor
.\.venv\Scripts\python.exe -m kotoba_lipsync benchmark input.wav
.\.venv\Scripts\python.exe -m scripts.validate_runtime input.wav
.\.venv\Scripts\python.exe -m pytest -q
powershell -NoProfile -File scripts/build_windows.ps1
```

配布版は `dist/KotobaLipSync/KotobaLipSync.exe` です。Python・モデル・依存ライセンス・日本語文書を同梱します。元音声、教師、CUDA/PyTorch、学習キャッシュは含めません。

## GitHubにあるもの・ローカルだけにあるもの

コード、採用・途中モデル、固定取得元、個別CC0選定、分割メタデータ、評価、実行記録を保存しています。新しいPCで全工程を1コマンドで完全再現する仕組みは未実装です。ローカルの `.pt`、音声、教師、特徴、段階ごとの追加インデックスはGitHubに含めません。選定結果は `reports/split_manifest_current.json` に保存し、出典の固定版と照合できるようにしています。

因果的な周波数平均補正の `--feature-mode cmn-v1` は実験用です。実装・テスト済みですが、この方式を使った学習・採用・精度確認はしていません。既定の `base` を採用モデルで使用しています。
