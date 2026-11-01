# 説明資料PDF

2026-11-01の提出用原稿は`content_submission.json`、生成処理は`build.swift`です。実測結果・条件比較・失敗分析・目標到達・限界・出典を4ページにまとめています。`content.json`と`microduck_report_draft.pdf`は以前の準備稿として残しています。

Macのプロジェクトルートで実行します。追加のPython依存は不要です。

```bash
swift -module-cache-path /private/tmp/microduck-swift-cache docs/report/build.swift --submission
```

出力は`output/pdf/microduck_report_submission.pdf`。本文は提出用ですが、この汎用ファイル名は提出規定の識別子付き名称ではありません。課題PDF原文の指定はアカウント名です。次の`ACCOUNT_NAME`を実際のアカウント名へ置き換えて生成します。

```bash
swift -module-cache-path /private/tmp/microduck-swift-cache docs/report/build.swift --submission --id ACCOUNT_NAME
```

出力名は`説明資料_PAI最終課題_ACCOUNT_NAME.pdf`。引数名は`--id`ですが、英数字・ハイフン・アンダースコアのアカウント名を渡せます。他の文字を含む名前は生成済みPDFをFinderで改名します。動画タイトルにはIDも可と明記されています。個人名のアカウント名をPDF名でIDへ代える扱いは、提出フォームの補足案内を確認します。本文には氏名・所属・メールを含めず、コード・動画URLはOmniCampusの提出欄へ記載する構成です。URLの提出情報は`docs/submission/finish_steps.md`を参照してください。

## 根拠と検証

- 学習：`outputs/review/baseline_training/`とBrev外部バックアップ内の各`run.json`、設定、checkpoint sidecar。
- 共通10試行：`outputs/comparison_final/`、`outputs/review/baseline_final/`、`outputs/review/low_speed_final/`。
- 目標到達：`outputs/review/baseline_1000_goal/`、`hybrid_goal_pilot/`、`hybrid_goal_validation/`、`hybrid_goal_fast_pilot/`、`hybrid_goal_fast_validation/`。
- 提出動画：`output/video/microduck-fast-validation-all.mp4`、152.24秒・全10試行・等速。

内容変更後は全ページを描画して確認します。生成コードは文章の見切れとページ下端を検査し、正常に完了してからPDFを置き換えます。

```bash
pdftoppm -scale-to 1400 -png output/pdf/microduck_report_submission.pdf tmp/pdfs/submission
pdftotext -layout output/pdf/microduck_report_submission.pdf tmp/pdfs/submission.txt
pdfinfo output/pdf/microduck_report_submission.pdf
```

PDFを編集する処理は学習ソースのハッシュ対象外です。PDF実体や評価データは公開用コードへ自動では同梱しません。
