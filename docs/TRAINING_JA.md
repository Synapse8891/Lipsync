# 学習を引き継ぐ手順

2026年10月24日時点。PowerShell用です。現在、こちらから動かしている学習・ラベル生成プロセスはありません。

## このPCでそのまま実行する場合

ターミナルに以下を順に入力します。

```powershell
cd C:\Users\harus\Downloads\Lipsync
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe scripts\run_training.py
```

最後の1コマンドで、次を順番に実行します。

1. 公開音声と教師モデルの確認・取得。すでにあるファイルは再取得しません。
2. 修正済みの前処理で音素ラベルを生成し、`cache/features_v2/` に保存。
3. 日本語1,800件・英語200件を学習候補として小型MLPを学習。日本語各180件・英語各40件を検証・テスト用に選択。長さ条件などで除外される録音があります。
4. 最良の検証値のモデルをint8で保存し、テスト評価。
5. サンプル音声でCPUベンチマークし、日本語の結果文書を生成。

このPCには元音声と教師の重みを取得済みです。**旧ラベルの再利用はできないため、日本語のラベル生成からやり直す必要があります。** この手順は旧 `cache/features/` を削除せず、新しい場所へ書き込みます。ラベル生成中の途中ファイルは1録音ずつ保存するため、中断後に同じコマンドを入力すると、署名が一致する完了分を再利用します。

CPUで動作します。処理時間はPCの性能と音声数に依存します。`Ctrl+C` で止められます。ニューラルモデルの学習途中からのエポック再開は未実装です。学習を再実行すると重みの初期化から始まります。

## 工程を分けて実行する場合

```powershell
cd C:\Users\harus\Downloads\Lipsync

# 取得と教師の前処理設定ファイルの準備
.\.venv\Scripts\python.exe scripts\download_training.py --ja-shards 3

# 修正済みラベルの生成。初回はここが比較的長くなります。
.\.venv\Scripts\python.exe scripts\prepare_training.py --ja-train 1800 --en-train 200 --ja-eval 180 --en-eval 40 --threads 4

# 小型モデルの学習と最終評価
.\.venv\Scripts\python.exe scripts\train.py --epochs 100 --hidden 192 96 --lookahead 3

# アプリの起動
.\.venv\Scripts\python.exe -m kotoba_lipsync gui
```

**`train.py` だけを最初に実行しないでください。** v2ラベルがなければエラーで止まるようにしています。v2ラベルがすべて生成済みなら、次回からは次のコマンドで学習以降を実行できます。

```powershell
.\.venv\Scripts\python.exe scripts\run_training.py --skip-download --skip-prepare
```

`--hidden 128 64` にするとモデルがさらに小さくなります。`--lookahead 0` は先読みなし、`3` は30 ms分待つ設定です。モデルを選び直す際は検証セットを使い、テスト結果を見て何度も選び直さないでください。

## 出力先

| ファイル | 内容 |
|---|---|
| `models/kotoba_int8.npz` | 新しく学習した推論用モデル |
| `reports/training_report.json` | 言語別一致率、発話中一致率、macro-F1、クラス別評価、学習履歴 |
| `reports/benchmark.json` | CPU処理時間、モデルサイズ、Windowsでのプロセスメモリ |
| `reports/RESULTS.md` | 自動生成された日本語の結果 |
| `reports/split_manifest.json` | 録音・話者・文章ハッシュと分割情報 |
| `data/provenance.json` | 取得元、固定版、ダウンロードファイルのSHA-256 |

一致率は**自動音素ラベルとの一致率**です。実際の唇や人手正解ラベルに対して80%以上を保証する値ではありません。特に自然な吸気・吐気は未検証です。

## テストとWindows実行ファイル化

```powershell
.\.venv\Scripts\python.exe -m pytest -q

# 学習完了後に実行。Pythonを同梱した配布フォルダを作ります。
powershell -NoProfile -File .\scripts\build_windows.ps1
```

配布フォルダは `dist/KotobaLipSync/`、起動はその中の `KotobaLipSync.exe` です。モデルだけではなくPython・数値計算ライブラリも含むため、アプリ全体の容量はモデル容量より大きくなります。ビルドには成功しましたが、現在の試行ビルドは有効なモデルを同梱していません。学習後に作り直してください。

Unity/VRChat/VMCの設定は [README](../README.md) を参照してください。

## 新しいPCでGitHubから取得した場合

Python 3.11とGitを入れたうえで、次のように実行します。非公開リポジトリなので、GitHubへの認証が必要です。

```powershell
git clone https://github.com/Synapse8891/Lipsync.git
cd Lipsync
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe scripts\run_training.py
```

厳密にこの作業時点のPythonパッケージ版へ揃えたい場合は、インストール前に `-m pip install -r requirements-lock.txt` を実行できます。
