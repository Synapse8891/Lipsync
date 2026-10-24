# Kotoba LipSync

日本語を中心に、英語も学習する軽量リップシンクです。マイク・音声ファイルから15口形と息の補助値を推定し、Unity UDP / VRChat OSC / VTuber向けVMCに出力します。GPU・音声認識API・クラウド接続は実行時に不要です。

**現在の状態（2026年10月25日）: 前処理修正後の5時間設定による学習が完了し、試作モデルを保存しました。** モデルは143,433 bytes・155,024パラメータです。テストの自動音素ラベル一致率は全体64.66%、日本語68.67%、英語53.99%。無音を除く全体の一致率は49.56%で、目標の80%以上には届いていません。実アバターでの見た目は未検証です。

- [ターミナルに入力する学習コマンド](docs/TRAINING_JA.md)
- [実施済み・未実施・問題点の記録](docs/WORK_LOG_JA.md)
- [現在の結果と検証範囲](reports/RESULTS.md)

5時間ほどの計算枠を使う場合は、`python scripts/run_training.py --profile five-hour` を使います。日本語8,000件・英語1,200件を目標にラベル生成を増やし、最大6候補を検証データで比較します。時間枠や早期終了で候補数は減る場合があります。[詳細](docs/TRAINING_JA.md#5時間の計算枠を使う設定)

**精度80%以上は、学習を回しただけでは保証しません。** `reports/training_report.json` は自動音素ラベルとの一致率を示します。人が確認した口形精度、カメラで計測した唇との一致率、自然な呼吸の識別精度とは異なります。実測値と制約は [学習結果](reports/RESULTS.md) を確認してください。

## 起動

**`Start_Kotoba.bat` をダブルクリック**すると、保存済みの試作モデルで画面を起動できます。モデル同梱のWindows配布版があれば、その実行ファイルを起動します。再学習する場合は [学習手順](docs/TRAINING_JA.md) を参照してください。

ソース版のみの場合、初回セットアップはPython 3.11を使います。ここには構築済みの `.venv` もあります。Windows配布版にはPythonが含まれているため、別途インストールは不要です。

1. マイクを選択します。
2. 出力先を「モニター」「Unity」「VRChat」「VTuber」から選びます。
3. 「マイクを開始」を押します。「停止する」でニュートラル値を送り終了します。

通常の会話が消える場合は無音しきい値を `-65` dBFS程度まで下げてください。周囲の音で口が動く場合は `-45` dBFS程度まで上げます。平滑化は初期値25 msです。データは録音・送信先として指定したUDPホストへの送信以外には保存しません。モニターではUDP送信もしません。

「WAVを解析・保存」はJSONLに口形、5母音、フォルマント、音量などを保存します。ファイルの再生は行いません。

## 推論の構成

- 16 kHz・モノラル、25 msの分析窓、10 ms刻み。
- 40次元log-mel、13次元MFCC、12次元LPC、F1〜F3推定、音量・ゼロ交差・スペクトル平坦度/重心/変化・帯域比・周期性・F0など、1フレーム78次元。
- 最新と10/20/40/80/120 ms前の特徴を連結。標準モデルは口形の遷移を判別するため30 ms分を待って出力します。ゼロ先読みモデルは `scripts/train.py --lookahead 0` で学習できます。
- 小型MLP。配布モデルの重みを出力チャネル別int8に量子化。実行時はfloat32に展開してCPUで計算します。整数専用演算や全体のメモリ1/4を意味するものではありません。
- PPの立ち上がりを速くした平滑化、固定音量の無音ゲート。
- 44.1/48 kHzマイクは境界を保持して16 kHzへ変換します。アンチエイリアス用に10 msをバッファします。

15口形は `sil PP FF TH DD kk CH SS nn RR aa E ih oh ou`、16番目は `breath` です。日本語の母音、閉鎖・破裂音、摩擦音、鼻音などをこの口形へ集約しています。これは文章の音声認識器や厳密な音素識別器ではありません。

`nasal` は `nn` 口形の強さであり、鼻腔の共鳴量を直接測定したものではありません。`breath` は /h/ 系の音を含む補助ラベルです。口形とは別の自然な吸気・吐気のラベル付き実録データによる検証は未実施です。フォルマントも短時間LPCの推定値で、無声子音・高い声・雑音では不安定になります。

## Unity

1. Unity Package Managerの「Add package from disk」で `unity/com.kotoba.lipsync/package.json` を選択。
2. GameObjectに `KotobaUdpReceiver` を付け、`face` に顔のSkinnedMeshRendererを設定。
3. 15口形モデルなら `Oculus15`、A/I/U/E/Oのモデルなら `FiveVowels` を選び、実際のブレンドシェイプ名へ変更。
4. Playにして、Kotobaの出力をUnity、ポート39541で開始。

受信はローカルPCに限定しています。他PCから受信したい場合だけ `allowRemoteHost` を有効にしてください。受信が0.3秒途絶えると口をニュートラルへ戻します。既存の表情コントローラーも同じ口形を操作している場合は、操作の競合を調整してください。

Python側が推論し、Unity側が口形を適用する構成です。Unity単体、スマートフォン単体、Quest単体のネイティブ推論実装ではありません。

## VRChat

標準の `Viseme` と `Voice` は読み取り専用のため、直接書き換えません。[VRChat公式仕様](https://creators.vrchat.com/avatars/animator-parameters/)

1. VRChat SDK3 Avatarsが入ったUnityプロジェクトに上記パッケージを追加。
2. `Tools > Kotoba LipSync > VRChat FX setup` を開く。
3. アバターのルート、顔メッシュ、口形のブレンドシェイプ名を指定。存在しない名前は空欄にします。
4. `Create and assign FX assets` を押すと、新しいFX・アニメーション・Expression Parametersを作成してDescriptorへ割り当てます。既存のFX・パラメータの元アセットは保持します。Descriptor変更はUndoできます。
5. Build & Testで確認してアバターを使用し、VRChatのOSCをONにします。
6. Kotobaの出力をVRChat、宛先127.0.0.1、ポート9000で開始。

`/avatar/parameters/Kotoba_{口形名}` の16個のfloat、`Kotoba_Enable` のboolを使います。同期する場合は129 bitを追加します。既存パラメータとの合計が256 bitを超えると作成を止めます。LipSyncはVisemeParameterOnlyへ設定し、標準機能による同じブレンドシェイプの書き込みを避けます。

他プレイヤーへの表情同期はVRChatのカスタムパラメータ同期に制限されます。ローカルと同じ10 ms更新が他人へ届く保証はありません。異常終了すると最終パラメータが残る場合があります。その場合はOSC送信の再起動・正常停止、またはアバターの再読み込みを行ってください。

このC#受信スクリプトをVRChatアバター内で実行する方式ではありません。外部PCで推論し、OSCとFXアニメーターで制御します。VRC SDK入り実アバターでの動作は未検証です。

## VTuber / VRM

VMC受信をサポートするアプリで外部ブレンドシェイプの受信を有効にし、ポートを合わせてください。初期値は39540です。

`/VMC/Ext/Blend/Val` に `A I U E O` と0〜1の値を送り、最後に `/VMC/Ext/Blend/Apply` を送ります。[VMC公式プロトコル](https://protocol.vmc.info/english)

VRM1の `aa ih ou ee oh` を直接受け取るアプリにはCLIの `--vrm1` を指定できます。通常は互換性のためVRM0名を使ってください。Live2D / VTube Studio専用APIとの直接接続は含みません。

## CLI

```powershell
.\.venv\Scripts\python.exe -m kotoba_lipsync devices
.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile vrchat --device 1
.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile vmc --port 39540
.\.venv\Scripts\python.exe -m kotoba_lipsync file input.wav --output outputs\visemes.jsonl
.\.venv\Scripts\python.exe -m kotoba_lipsync file input.wav --profile unity --realtime
.\.venv\Scripts\python.exe -m kotoba_lipsync benchmark input.wav
```

UDP JSONの `weights` は15口形の固定順、`vrm` はA/I/U/E/Oの順です。`breath` は別フィールドです。`timestamp` は分析窓中心の時刻、`inference_ms` はモデルと平滑化部分のみです。ベンチマークは特徴抽出を含めて測定します。

## 学習の再現

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe scripts\download_training.py --ja-shards 3
.\.venv\Scripts\python.exe scripts\prepare_training.py --ja-train 1800 --en-train 200 --ja-eval 180 --en-eval 40
.\.venv\Scripts\python.exe scripts\train.py --epochs 100 --hidden 192 96
.\.venv\Scripts\python.exe -m pytest -q
```

データはCC0のMozilla Common Voice 17.0日本語・英語です。ダウンロードは約1.9 GBで、元音声・教師モデル・中間特徴は `data/` と `cache/` に保存します。学習データへのクレジット表示義務はありません。出典・版・ハッシュは監査用に記録しています。製品の動画や画面にコーパスのクレジットを追加する必要はありません。

修正済みラベルは `cache/features_v2/` に作ります。旧 `cache/features/` は使いません。教師ごとの `preprocessor_config.json`、教師の重み・語彙・口形マッピング・特徴設定を識別する署名を保存し、異なるキャッシュの混在を防ぎます。日本語教師は `do_normalize=false`、英語教師は `true` です。取得元のリビジョンは `configs/training_sources.json` に固定しています。

日本語HuBERT CTC / 英語wav2vec2 CTCのApache-2.0モデルは、学習用の音素ラベルを自動生成するためだけに使います。実行時には使いません。CTCスパイク間の中点でラベルを展開しており、音素境界は人が修正したものではありません。教師の誤認識も引き継ぐため、評価値は教師との一致率です。

分割は音声・話者・文章ハッシュの重複を排除します。正規化は学習用データのみで計算し、検証用の発話中一致率とmacro-F1でエポックを選択します。最終int8モデルをテストで評価します。テスト値を見て何度もモデルを選ぶと評価が甘くなるため、追加の改良には新しい未使用テストセットを用意してください。

## ライセンス

このプロジェクトの新規コード・学習した小型モデルは [CC0-1.0](LICENSE) で提供します。商用利用・改変・再配布にプロジェクトのクレジットは不要です。依存ライブラリ・学習用教師モデルはそれぞれのライセンスです。ソフトウェアを配布する場合のライセンス文書保持については [第三者情報](THIRD_PARTY_NOTICES.md) を確認してください。
