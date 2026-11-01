# 動作確認用の動画

Macのプロジェクトルートで生成します。Python標準ライブラリー、FFmpeg/ffprobe、日本語システムフォントを使います。学習・シミュレーションの再実行は不要です。

```bash
python3 docs/video/build_review.py
```

入力は`outputs/review/hybrid_goal_pilot/`、`hybrid_goal_fast_pilot/`、`hybrid_goal_fast_validation/`の検証済みJSON/MP4です。入力動画のSHAを評価記録と照合してから編集します。

| 出力 | 内容 |
| --- | --- |
| `output/video/microduck-speed-comparison.mp4` | 39.00秒。共通の開発seed 300〜302を左右に同時表示。左が変更前、右が速度変更版。両方等速で、先に終了した側は終了フレームの静止と明記。全3試行を含む。 |
| `output/video/microduck-fast-validation-all.mp4` | 152.24秒。速度変更版の未使用seed 500〜509を全10試行収録。各試行を最初から最後まで等速で表示し、間の結果画面でリセットの区切りを示す。 |
| `output/video/review_manifest.json` | 入出力ハッシュ、長さ、編集内容、各区間の開始時刻、生成スクリプトSHA。 |
| `output/video/review_qa.json` | デコード・仕様・代表フレームの確認結果。 |

全10試行の動画は、元映像の記録用字幕を残し、シーンの上下の余白だけを切り取って拡大しています。範囲は640×480の元映像に対し、シーン部分がx=0、y=160、幅640、高さ280。ロボット・目標が映る横幅は保持します。時間の短縮、倍速、試行途中の省略、ロボットの動きの生成は行っていません。無音のH.264 / yuv420p / 1280×720 / 50fpsです。

速度の比較は同じ開発3試行の平均9.31→8.01秒（約14%短縮）。速度変更版の独立した10試行は10/10成功、停止完了まで平均11.80秒でした。seed・初期条件が違うため、この2種類の平均時間は直接比較しません。

どちらも確認用で、公開・提出は未実施です。全10試行の動画は60〜180秒という長さの条件を満たしますが、最終提出時の構成・タイトル・識別子・URLは別途確定します。Macでは動画ファイルをダブルクリックするか、次で開けます。

```bash
open output/video/microduck-speed-comparison.mp4
open output/video/microduck-fast-validation-all.mp4
```
