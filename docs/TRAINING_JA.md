# 学習を引き継ぐ手順

2026年10月24日時点。PowerShell用です。現在、こちらから動かしている学習・ラベル生成プロセスはありません。

**2026年10月25日追記:** ユーザーによる5時間設定の実行が正常終了し、`models/kotoba_int8.npz` を保存しています。試すだけなら再学習は不要で、`Start_Kotoba.bat` を起動できます。以下は再学習・別環境での再現用です。実測結果は [RESULTS.md](../reports/RESULTS.md) を参照してください。

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

## 5時間の計算枠を使う設定

```powershell
cd C:\Users\harus\Downloads\Lipsync
.\.venv\Scripts\python.exe scripts\run_training.py --profile five-hour
```

この設定では、同じデータを5時間反復するのではなく、次を行います。

- 日本語の学習用データを最大8シャードまで取得し、学習候補を日本語8,000件・英語1,200件へ拡大。条件に合う録音が足りない場合は、利用可能な件数を使います。
- 検証・テスト用に、日本語はそれぞれ300件、英語はそれぞれ100件を選択。学習との話者・文章・録音重複を排除します。
- 学習特徴を小分けに読み込み、大量の特徴をメモリへ一括展開しません。
- 隠れ層 `128/64`、`192/96`、`256/128` とシード42/73の組み合わせを最大6候補まで比較。最大のモデルでも155,024パラメータです。各候補は最大100エポック、検証値が18エポック改善しなければ終了します。
- 量子化した候補を検証データで選び、選ばれたモデルだけを最後にテストします。テスト値を使った候補選択はしません。

**5時間は取得・前処理・学習・評価を合わせた計算予算です。必ず5時間動かす指定ではありません。** ラベル生成には開始から最大3時間30分を割り当て、残りを学習比較などに使います。評価用データを先に準備し、その後は日本語・英語の学習データを交互に増やします。前処理の期限に達した場合、完成分の一覧を保存して、用意できたデータで比較します。学習用データがまだ準備できていなければ停止し、キャッシュは次回へ残します。

時間の確認はネットワークI/O、1録音、学習バッチなどの区切りで行います。処理中のI/Oや最終評価の分だけ5時間を超える場合があります。PCのスリープ中は学習が進みませんが、経過時間には含まれます。

**2026年10月25日に実データでの運転が完了しました。** 全体で約4時間58分、全6候補を比較しています。自動ラベルとのテスト一致率は64.66%で、80%以上には届いていません。呼吸の専用データを追加する設定ではありません。

実行せずに内容だけ確認するには、次を使います。

```powershell
.\.venv\Scripts\python.exe scripts\run_training.py --profile five-hour --dry-run
```

5時間設定でラベル生成が済んでいれば、次回は次のように比較だけを実行できます。通常設定の `index.json` だけでは代用できません。

```powershell
.\.venv\Scripts\python.exe scripts\run_training.py --profile five-hour --skip-download --skip-prepare
```

状態は `reports/run_status.json`、比較の途中モデルと履歴は `cache/search_models/` に保存します。5時間設定の一覧は `cache/features_v2/index_five_hour.json` で、通常設定の一覧を上書きしません。最終モデルと評価結果の保存先は通常設定と同じです。時間や中断によって、全6候補が完了しない場合があります。

## 通常設定の工程を分けて実行する場合

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
