"""Render concise Japanese model/validation documentation from measured JSON."""
import json
from pathlib import Path
import platform

ROOT=Path(__file__).resolve().parents[1]

def main():
    r=json.loads((ROOT/"reports/training_report.json").read_text(encoding="utf-8"))
    b=json.loads((ROOT/"reports/benchmark.json").read_text(encoding="utf-8"))
    lines=["# 学習・評価結果", "", "**80%以上の実音声リップシンク精度は未確認です。** 下記は自動CTC音素ラベルに対する一致率であり、人が確認した音素・実際の唇の正解率ではありません。", "", "## 実施した学習", "", "CC0のMozilla Common Voice 17.0、日本語を主・英語を補助として、小型MLPを実際に学習しました。発話・話者・文章のハッシュを学習・検証・テスト間で分離しています。", "", "| 分割 | 言語 | 発話数 | 話者数 | 音声時間 |", "|---|---|---:|---:|---:|"]
    for key,value in r["data_summary"].items():
        lang,split=key.split("/")
        lines.append(f"| {split} | {lang} | {value['recordings']} | {value['speakers']} | {value['seconds']/60:.1f}分 |")
    lines += ["", f"ネットワーク: 468 → {' → '.join(map(str,r['training_config']['hidden']))} → 16。パラメータ {r['parameter_count']:,}、重み保存int8、配布モデル {r['model_bytes']:,} bytes（{r['model_bytes']/1024:.1f} KiB）。CPU演算時には重みをfloat32に展開します。", "", f"最大{r['training_config']['epochs_max']}エポック、検証値で採用したエポックは{r['best_epoch']}。実行したエポック数は{len(r['history'])}。乱数シード{r['training_config'].get('seed',42)}。テストでモデル選択はしていません。", "", "## 未使用話者での最終int8モデル評価", "", "| 対象 | 全フレーム一致率 | 発話中の一致率 | macro-F1 | 多数クラス基準 |", "|---|---:|---:|---:|---:|"]
    for lang in ("all","ja","en"):
        s=r["test"][lang]
        lines.append(f"| {lang} | {s['accuracy']:.2%} | {s['speech_accuracy']:.2%} | {s['macro_f1']:.3f} | {s['majority_class_baseline']:.2%} |")
    lines += ["", "発話中は正解ラベルがsil以外のフレームです。macro-F1はクラスごとのF1の平均です。評価は実行時の平滑化前の量子化MLPのargmaxです。音素から口形への規則的な対応と、自動生成された境界の誤差を含みます。フレーム単位のラベル一致率が、アバターの見た目の良さや音声認識の正解率を保証するものではありません。", "", "| 口形 | Precision | Recall | F1 | テストフレーム数 |", "|---|---:|---:|---:|---:|"]
    for label in r["labels"] if "labels" in r else ("sil","PP","FF","TH","DD","kk","CH","SS","nn","RR","aa","E","ih","oh","ou","breath"):
        s=r["test"]["all"]["per_class"][label]
        lines.append(f"| {label} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1-score']:.3f} | {int(s['support'])} |")
    lines += ["", "`breath`は/h/系を含む補助クラスで、自然な吸気・吐気の正解ラベルではありません。鼻音はnn口形への対応であり、鼻腔の共鳴の直接測定ではありません。", "", "## CPU実測", "", f"測定環境: {platform.platform()} / {platform.processor()}。Python {platform.python_version()}。GPUは使用しません。", "", f"- 音声 {b['audio_seconds']:.2f}秒を{b['elapsed_seconds']:.3f}秒で処理。実時間係数 {b['real_time_factor']:.4f}（1未満で実時間より高速）。", f"- 10 msフレームあたり、特徴抽出＋推論 p50 {b['frame_ms_p50']:.3f} ms、p95 {b['frame_ms_p95']:.3f} ms、p99 {b['frame_ms_p99']:.3f} ms。", f"- float32展開したモデル重み・バイアス {b['model_float_weights_bytes']:,} bytes。"]
    if b.get("process_working_set_bytes"):
        lines.append(f"- CLIプロセス全体のワーキングセット {b['process_working_set_bytes']/1024**2:.1f} MiB。")
    lines += [
        "",
        f"25 ms分析窓、先読み{b['lookahead_ms']} ms、44.1/48 kHz入力でのリサンプラーバッファ10 ms、初期平滑化25 ms。マイク入力から描画までの遅延は未測定です。",
        "",
        "## 接続・検証範囲",
        "",
        "- ストリーミング分割不変性、先読み上限、44.1/48 kHzリサンプリングの連続性、int8出力誤差、OSCバイナリ、UDP送受信、停止時ニュートラル、話者・文章重複の拒否をテスト。",
        "- CLIで入力デバイスの列挙を確認。",
        "- Unity 6000.3.9f1のDLLを参照したC#コンパイルは成功。Unity Editorの起動はライセンス未認証で終了。",
        "- VRChat SDK入り実アバター、VMC受信アプリ、実マイクからの見た目・聴感・他プレイヤー同期は未検証。",
        "",
        "## 80%以上を確認するために残る作業",
        "",
        "CC0で利用できる実録の自然呼吸・ささやき・破裂・鼻音と、時刻を確認した口形・音素ラベルの追加が必要です。未学習話者の日本語・英語を用い、録音環境やマイクが異なる未使用データで発話中一致率・macro-F1・呼吸のprecisionとrecallを評価します。",
        "",
        "測定データは [training_report.json](training_report.json)、[benchmark.json](benchmark.json)、[split_manifest.json](split_manifest.json) を参照してください。",
    ]
    (ROOT/"reports/RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":main()
