# Kotoba LipSync

日本語を主対象としたリアルタイムのリップシンクです。マイクや音声ファイルから15種類の口形と呼吸の強さを推定し、Unity（UDP）とVRChat（OSC）に送信します。VTuber向けのVMCには5母音を送信します。推論はローカルのCPUで実行します。

## 評価結果

| 項目 | 採用モデルの結果 |
| --- | --- |
| 日本語の自動音素ラベル一致率（未使用200録音） | 75.06%（無音を除くと65.81%） |
| 英語の自動音素ラベル一致率（未使用60録音） | 63.14% |
| 自然呼吸の検出 | 9録音中8録音を検出、陰性268録音中5録音で誤反応 |
| モデルサイズ | 167,276バイト（約163 KiB） |
| 特徴抽出＋CPU推論（10 ms分の入力あたり） | 中央値0.690 ms、95パーセンタイル0.762 ms |

一致率の基準は自動CTC音素ラベルです。呼吸は投稿者の説明に基づくクリップ単位の評価です。実アバター上の見た目とマイク入力から描画までの総遅延は未計測です。[評価方法・測定環境](reports/RESULTS.md)

評価レポートの整理に Codex を使用しました。

## CLIで試す

リポジトリのルートで以下を実行します。同梱の学習済みモデルを使います。Windows PowerShell と Python 3.11 の例です。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m kotoba_lipsync file input.wav --output outputs\visemes.jsonl
```

`input.wav` を手元の音声ファイルに置き換えると、口形、5母音、呼吸、フォルマント、音量をJSONLで確認できます。マイクを使うには `devices` で入力番号を確認し、`mic --device 1 --profile monitor` のように指定します。モニターは端末に表示し、接続先へ数値を送る場合は `unity`、`vrchat`、`vmc` を選びます。マイク入力は Ctrl+C で停止します。

## 推論の構成

- 16 kHz・モノラル、25 msの分析窓、10 ms刻み。
- 40次元log-mel、13次元MFCC、12次元LPC、F1〜F3推定、音量・ゼロ交差・スペクトル平坦度/重心/変化・帯域比・周期性・F0など、1フレーム78次元。
- 生の78次元とフレーム内で平均を引いた40次元log-mel、合計118次元を入力し、96ユニット×2層GRUと小型の時間特徴分岐で推論します。
- GRUの先読み100 msと確率補正の20 ms、合計120 ms分を待って出力します。精度改善と引き換えに旧MLPの30 msより長くなっています。
- 配布モデルの重みを出力チャネル別int8に量子化。実行時はfloat32に展開してCPUで計算します。整数専用演算や全体のメモリ1/4を意味するものではありません。
- PPの立ち上がりを速くした平滑化、固定音量の無音ゲート。
- 44.1/48 kHzマイクは境界を保持して16 kHzへ変換します。アンチエイリアス用に10 msをバッファします。

15口形は `sil PP FF TH DD kk CH SS nn RR aa E ih oh ou`、内部の16番目は/h/系の `HH` です。日本語の母音、閉鎖・破裂音、摩擦音、鼻音などをこの口形へ集約しています。これとは別に自然呼吸の二値ヘッドを持ちます。文章の音声認識器や厳密な音素識別器ではありません。

`nasal` は `nn` 口形の強さであり、鼻腔の共鳴量を直接測定したものではありません。出力の `breath` は独立した自然呼吸の補助値です。吸気と吐気の区別、ささやき専用の精度は未検証です。フォルマントも短時間LPCの推定値で、無声子音・高い声・雑音では不安定になります。

口形の更新は100回/秒です。先読み120 ms、分析窓中心から12.5 ms、平滑化の時定数10 msに加え、マイク・ドライバー・通信・描画にも時間がかかります。

## Unity

1. Unity Package Managerの「Add package from disk」で `unity/com.kotoba.lipsync/package.json` を選択。
2. GameObjectに `KotobaUdpReceiver` を付け、`face` に顔のSkinnedMeshRendererを設定。
3. 15口形モデルなら `Oculus15`、A/I/U/E/Oのモデルなら `FiveVowels` を選び、実際のブレンドシェイプ名へ変更。
4. UnityでPlayし、`.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile unity` で既定ポート39541へ送信。

任意の `breathBlendShape` を設定すると、呼吸の補助値も適用できます。VMC出力は5母音のみで、呼吸の専用送信は含みません。

受信はローカルPCに限定しています。他PCから受信したい場合だけ `allowRemoteHost` を有効にしてください。受信が0.3秒途絶えると口をニュートラルへ戻します。既存の表情コントローラーも同じ口形を操作している場合は、操作の競合を調整してください。

Python側が推論し、Unity側が口形を適用する構成です。Unity単体、スマートフォン単体、Quest単体のネイティブ推論実装ではありません。

## VRChat

標準の `Viseme` と `Voice` は読み取り専用のため、直接書き換えません。[VRChat公式仕様](https://creators.vrchat.com/avatars/animator-parameters/)

1. VRChat SDK3 Avatarsが入ったUnityプロジェクトに上記パッケージを追加。
2. `Tools > Kotoba LipSync > VRChat FX setup` を開く。
3. アバターのルート、顔メッシュ、口形のブレンドシェイプ名を指定。存在しない名前は空欄にします。
4. `Create and assign FX assets` を押すと、新しいFX・アニメーション・Expression Parametersを作成してDescriptorへ割り当てます。既存のFX・パラメータの元アセットは保持します。Descriptor変更はUndoできます。
5. Build & Testで確認してアバターを使用し、VRChatのOSCをONにします。
6. `.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile vrchat` で既定の宛先127.0.0.1、ポート9000へ送信。

`/avatar/parameters/Kotoba_{口形名}` の16個のfloat、`Kotoba_Enable` のboolを使います。同期する場合は129 bitを追加します。既存パラメータとの合計が256 bitを超えると作成を止めます。LipSyncはVisemeParameterOnlyへ設定し、標準機能による同じブレンドシェイプの書き込みを避けます。

他プレイヤーへの表情同期はVRChatのカスタムパラメータ同期に制限されます。ローカルと同じ10 ms更新が他人へ届く保証はありません。異常終了すると最終パラメータが残る場合があります。その場合はOSC送信の再起動・正常停止、またはアバターの再読み込みを行ってください。

このC#受信スクリプトをVRChatアバター内で実行する方式ではありません。外部PCで推論し、OSCとFXアニメーターで制御します。VRC SDK入り実アバターでの動作は未検証です。

## VTuber / VRM

VMC受信をサポートするアプリで外部ブレンドシェイプの受信を有効にし、ポートを合わせてください。初期値は39540です。

`/VMC/Ext/Blend/Val` に `A I U E O` と0〜1の値を送り、最後に `/VMC/Ext/Blend/Apply` を送ります。[VMC公式プロトコル](https://protocol.vmc.info/english)

VRM1の `aa ih ou ee oh` を直接受け取るアプリにはCLIの `--vrm1` を指定できます。通常は互換性のためVRM0名を使ってください。Live2D / VTube Studio専用APIとの直接接続は含みません。

## CLIコマンド

```powershell
.\.venv\Scripts\python.exe -m kotoba_lipsync devices
.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile vrchat --device 1
.\.venv\Scripts\python.exe -m kotoba_lipsync mic --profile vmc --port 39540
.\.venv\Scripts\python.exe -m kotoba_lipsync file input.wav --profile unity --realtime
.\.venv\Scripts\python.exe -m kotoba_lipsync benchmark input.wav --output outputs\benchmark.json
```

UDP JSONの `weights` は15口形の固定順、`vrm` はA/I/U/E/Oの順です。`breath` は別フィールドです。`timestamp` は分析窓中心の時刻、`inference_ms` はモデルと平滑化部分のみです。ベンチマークは特徴抽出を含めて測定します。

## 学習の再現

採用GRUの構成・コマンド・学習履歴は [追加学習手順](docs/TRAINING_CURRENT_JA.md) にまとめています。開発中の候補モデルの比較は [85%に向けた実験](docs/ACCURACY_85_JA.md) を参照してください。

以下は旧MLPの再現例です。既定モデルを書き換えるため、別のチェックアウトで実行してください。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe scripts\download_training.py --ja-shards 3
.\.venv\Scripts\python.exe scripts\prepare_training.py --ja-train 1800 --en-train 200 --ja-eval 180 --en-eval 40
.\.venv\Scripts\python.exe scripts\train.py --epochs 100 --hidden 192 96
.\.venv\Scripts\python.exe -m pytest -q
```

音声はCC0のMozilla Common Voice 17.0日本語・英語と、個別のCC0表示・ハッシュを確認した呼吸・環境音です。現在の学習用発話は日本語7,997件・英語2,800件。元音声・教師モデル・中間特徴は `data/` と `cache/` に保存し、GitHub・配布版には含めません。学習音声のクレジット表示は不要です。混合ライセンスのカタログ全体をCC0とは扱っていません。監査用出典と依存ソフトの文書は保持します。

修正済みラベルは `cache/features_v2/` に作ります。旧 `cache/features/` は使いません。教師ごとの `preprocessor_config.json`、教師の重み・語彙・口形マッピング・特徴設定を識別する署名を保存し、異なるキャッシュの混在を防ぎます。日本語教師は `do_normalize=false`、英語教師は `true` です。取得元のリビジョンは `configs/training_sources.json` に固定しています。

日本語HuBERT CTC / 英語wav2vec2 CTCのApache-2.0モデルは、学習用の音素ラベルを自動生成するためだけに使います。実行時には使いません。CTCスパイク間の中点でラベルを展開しており、音素境界は人が修正したものではありません。教師の誤認識も引き継ぐため、評価値は教師との一致率です。

分割は音声・話者・文章ハッシュの重複を排除します。正規化は学習用データのみで計算し、検証用の発話中一致率とmacro-F1でエポックを選択します。最終int8モデルをテストで評価します。テスト値を見て何度もモデルを選ぶと評価が甘くなるため、追加の改良には新しい未使用テストセットを用意してください。

## ライセンス

このプロジェクトの新規コード・学習した小型モデルは [CC0-1.0](LICENSE) で提供します。商用利用・改変・再配布にプロジェクトのクレジットは不要です。依存ライブラリ・学習用教師モデルはそれぞれのライセンスです。ソフトウェアを配布する場合のライセンス文書保持については [第三者情報](THIRD_PARTY_NOTICES.md) を確認してください。
