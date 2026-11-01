# Brev停止前の外部バックアップ

実行する場所を分けます。`outputs`全体のバックアップには、標準・低速条件の全チェックポイント、optimizer等を含む保存状態、設定、学習ログ、評価ログと動画が入ります。既存のbackupコマンドは自作コード・設定・tests・docs・licenses・assetsも`project/`に保存します。仮想環境やCUDA自体はコピーせず、復元時に固定コミットとlockから構築し直します。

## 1. MacからBrevへ作成処理を送る

学習・評価が終了した状態で、次をMacのターミナルから実行します。引用符内の処理だけがBrev上で実行されます。`uv.lock`と`pyproject.toml`はBrevのvendor内にあり、上流を取得していないMacのvendorにはありません。`set -e`で依存ファイルの取得に失敗したらアーカイブ作成へ進まないようにします。

```bash
ssh microduck '
set -eu
cd /home/ubuntu/workspace/advanced1
mkdir -p outputs/backup_dependencies
cp vendor/microduck_rl/uv.lock vendor/microduck_rl/pyproject.toml outputs/backup_dependencies/
python3 scripts/run.py backup outputs --output ../advanced1-backups/advanced1-20261031.tar.gz
'
```

出力は`/home/ubuntu/workspace/advanced1-backups/advanced1-20261031.tar.gz`と`.tar.gz.json`です。バックアップ先を対象のoutputsの外に置き、アーカイブ自身の混入を避けます。既存アーカイブは上書きしないため、再作成が必要な場合は別名を指定し、後の取得・検証でも同じ名前にします。

## 2. Macで取得

Macのプロジェクトルートで実行します。`/private/tmp`ではなく、Git対象外の`outputs/backups/brev/`へ保存します。

```bash
mkdir -p outputs/backups/brev
rsync -avP microduck:/home/ubuntu/workspace/advanced1-backups/ outputs/backups/brev/
```

rsyncが終了するまでBrevを停止しません。`-P`により中断時の部分ファイルを保持し、同じコマンドで転送を再開できます。

## 3. Macで照合

```bash
python3 docs/backup/verify_external.py outputs/backups/brev/advanced1-20261031.tar.gz
```

全アーカイブ内ファイルのSHAと、Brev作成時のアーカイブSHAを検査します。さらに、実測記録から固定した次の3モデルの重み、対応JSON、学習時のソース、保存したuv.lockを照合します。

| モデル | 使用目的 |
| --- | --- |
| `outputs/baseline/model_1000.pt` | 目標到達の移動用（1001更新時点） |
| `outputs/baseline/final.pt` | 停止用・標準条件の最終モデル |
| `outputs/low_speed/final.pt` | 低速条件の比較用最終モデル |

`verification: passed`と3件の`required_checkpoints`の成功を確認します。結果は`.tar.gz.local-verification.json`に残ります。Macでの照合成功後に、Brevのインスタンスを停止します。削除は行いません。この確認はバイト単位のコピーと構成の検証であり、MacでGPU推論を再実行した検証ではありません。

## 復元

アーカイブは非公開の保管用です。GitHubに巨大な重み・全ログをそのまま入れません。将来、Linux x86_64のNVIDIA GPU環境で、信頼する検証済みアーカイブを新しい空ディレクトリへ展開します。

```bash
mkdir restored-advanced1
tar -xzf advanced1-20261031.tar.gz -C restored-advanced1
mv restored-advanced1/run restored-advanced1/project/outputs
cd restored-advanced1/project
python3 scripts/bootstrap.py
```

`project/`が復元後のプロジェクトルートです。元の`outputs`全体が`run/`として保存されているため、上記で元の配置へ戻します。bootstrapは固定上流SHAから再取得し、Python 3.12の環境を再構築するのでインターネット接続が必要です。保存した`outputs/backup_dependencies/uv.lock`は元のlockの照合用で、上流設定を無断で置き換えるためには使いません。

元の動画に使ったモデルは`outputs/baseline/model_1000.pt`と`outputs/baseline/final.pt`です。`python3 docs/diagnostics/hybrid_goal.py --fast-validation --output outputs/restored_fast_validation`で再評価できます。元の評価先はバックアップに含まれるので、新しい出力先を指定します。学習を再開する場合は`train --resume`で累計更新数と新しい出力先を指定します。復元後のGPU動作と、再開後に同一軌跡となることは未検証です。
