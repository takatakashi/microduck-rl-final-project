````markdown
# Microduckの強化学習・条件比較・目標到達

Physical AI応用1の最終課題。MicroduckをMuJoCo Warp / mjlab上でPPO学習し、標準・低速条件を比較した。自分で学習した標準モデルの中間時点を移動、最終時点を停止に使うルール制御を追加し、指定地点への移動・停止を検証した。

## 実測結果

- 両条件ともscratch初期化、seed 42、1024環境、4000更新、98,304,000環境ステップ。BrevのL40S 1枚を使用。
- 標準の実時間は約51分50秒、低速条件は約52分7秒。変更は前後速度の学習範囲を±0.4から±0.2 m/sへ狭めた1項目。
- 最終モデル同士の共通10試行では両方30秒完走・転倒0。ただし前進指令中もほぼ静止し、低速条件の歩行改善は確認できなかった。
- 移動用model_1000（1001更新）と停止用final（4000更新）の切り替えで、未使用seed 400〜409は10/10成功。
- 接近速度の距離係数を0.8から1.2へ変更。共通の開発seed 300〜302で停止完了時間9.31→8.01秒、前進中の実速度0.139→0.188 m/s、成功は両方3/3。
- 速度変更版を固定した未使用seed 500〜509も10/10成功・転倒0。平均11.80秒、最終目標誤差11.61 cm。初期条件の違う試行の平均時間を直接比較しない。

目標は世界座標(0.6, 0.0) m。半径0.15 m以内で、速度0.05 m/s以下・旋回速度の絶対値0.15 rad/s以下を2秒維持すると成功。位置・姿勢はシミュレーターの真値を使用する。目標到達とモデル切り替えはルール制御、関節指令の生成はPPOで学習した部分である。実機への転用や任意目標への一般化は未検証。

## 固定環境

Linux x86_64、NVIDIA GPU、Python 3.12、uv、Git、FFmpeg/ffprobeが必要。学習時はPython 3.12.16、mjlab 1.3.0、Warp 1.12.0、PyTorch 2.9.1、rsl-rl 5.0.1を使用した。

上流: https://github.com/pollen-robotics/microduck_rl

採用SHA: `53b8971b61baf5b7f3c16d135dd7cac37623de4b`

```bash
python3 scripts/bootstrap.py
python3 scripts/run.py doctor --gpu --persistent-workspace /home/ubuntu/workspace
```

学習環境はvendor側の仮想環境を使い、scripts/run.pyが自動的にそのPythonへ切り替える。reproducibility/のlockとpyprojectは学習時の照合用コピーで、bootstrapが使用する固定上流のlockを置き換えない。

## 学習・比較・目標到達

```bash
python3 scripts/run.py train --config configs/experiments/baseline.json --output outputs/baseline
python3 scripts/run.py train --config configs/experiments/low_speed.json --output outputs/low_speed
python3 scripts/run.py evaluate --checkpoint outputs/baseline/final.pt --output outputs/baseline_eval --video
python3 scripts/run.py evaluate --checkpoint outputs/low_speed/final.pt --output outputs/low_speed_eval --video
python3 scripts/run.py compare outputs/baseline_eval/evaluation.json outputs/low_speed_eval/evaluation.json --output outputs/comparison
python3 docs/diagnostics/hybrid_goal.py --fast-validation
```

既存の出力先は上書きしない。再実行は--outputで新しいディレクトリを指定する。保存済み重みはoutputs/baseline/model_1000.pt、outputs/baseline/final.pt、outputs/low_speed/final.ptと対応する同名JSONを使う。このリポジトリには学習済み重みを同梱していないため、clone直後に評価だけを実行することはできない。上記の学習手順で重みを生成してから評価する。再学習で動画と完全に同じ軌跡になることは保証しない。

再開時の更新数は追加数ではなく累計更新数。ソース・依存・設定の互換性を検査する。完全に同じ軌跡の復元は保証しない。例えば標準の中間モデルから4000更新まで再開する場合は次を使う。

```bash
python3 scripts/run.py train --config configs/experiments/baseline.json --resume outputs/baseline/model_1000.pt --total-iterations 4000 --output outputs/baseline_resumed
```

ONNX書き出しと元モデルとの出力照合はGPU環境で次を使う。実際に`passed`となった対象モデルと`.parity.json`を対応付ける。短時間試験の照合結果を別モデルの合格として扱わない。

```bash
python3 scripts/run.py export --checkpoint outputs/baseline/model_1000.pt --output outputs/baseline/model_1000.onnx
```

## GPU不要の確認

Python 3.12で実行する。

```bash
python3 scripts/run.py validate
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s docs/diagnostics -p 'test_*.py' -v
```

## 再利用と独自部分

公式の機体モデル・観測・行動順序・BAMアクチュエーター・報酬・PPO・runnerを再利用した。独自部分は実験管理、条件比較、失敗分析、目標到達とモデル切り替え、記録・録画・互換性検査である。公式配布の学習済み重みは今回のタスク実行には使っていない。

出典と資産の扱いはTHIRD_PARTY_NOTICES.md、licenses/UPSTREAM_MODEL_NOTICE.md、assets/manifest.jsonを参照。コードと3Dモデルのライセンスを混同しない。
````
