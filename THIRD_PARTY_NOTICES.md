# 出典と再利用範囲

## Pollen Robotics / microduck_rl

- 原著者: Pollen Roboticsおよび上流の各著作権者。
- リポジトリ: https://github.com/pollen-robotics/microduck_rl
- 採用コミット: [53b8971b61baf5b7f3c16d135dd7cac37623de4b](https://github.com/pollen-robotics/microduck_rl/commit/53b8971b61baf5b7f3c16d135dd7cac37623de4b)
- 再利用: 平地歩行環境、観測と関節の順序、BAMアクチュエーター、ランダム化、報酬、PPO設定、runner、正規化付きONNX exporter、ロボットXMLとメッシュ。
- 上流コードは変更せず`vendor/microduck_rl`に取得。LICENSEと全ファイルの著作権表示を保持する。現時点ではローカル未取得であり、上流ファイルを同梱したとはしていない。
- 上流コードのライセンスはApache-2.0。STLモデルは[上流のモデル表示](https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE-MODELS.md)でCC BY-NC-SA 4.0。[参照記録](licenses/UPSTREAM_MODEL_NOTICE.md)に採用コミットとの時期の違いを記録している。コードのライセンスをSTLへ一律適用しない。
- 取得した全ファイルの出典とSHA-256は`assets/manifest.json`。公式学習済み重みは取得・同梱していない。

## その他の学習依存

MuJoCo（Google DeepMind等）、MuJoCo Warp（Google DeepMind / NVIDIA等）、mjlab（MuJoCo Lab等）、rsl_rl（ETH Zurich / NVIDIA等）、PyTorch、BAM（Rhoban等）、ONNX / ONNX Runtime等を上流のuv.lockに従って使用する。各依存のLICENSEと著作権表示は各配布物に従う。本プロジェクト独自の許諾で置き換えない。

API確認に参照した公式ソース:

- [mjlab 1.3.0 train](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/train.py)
- [mjlab runnerの保存・復元](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/rl/runner.py)
- [rsl_rl 5.0.1の学習ループ](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/runners/on_policy_runner.py)
- [公式Microduck exporter](https://github.com/pollen-robotics/microduck_rl/blob/53b8971b61baf5b7f3c16d135dd7cac37623de4b/src/mjlab_microduck/export.py)

## 今回追加した部分

`microduck_project/`、`scripts/bootstrap.py`、`scripts/run.py`、`configs/`、`tests/`と課題用ドキュメント。固定ソースの取得、実験差分、公式runnerを呼ぶ学習記録とcheckpoint検証、ルールによる上位制御、物理指標での比較、録画、ONNX照合、バックアップを追加した。学習アルゴリズムとロボット観測を新規実装したものではない。

課題PDF原本は再配布しない。提出者の識別情報・認証情報はこの通知に含めない。
