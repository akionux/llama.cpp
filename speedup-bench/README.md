# speedup-2026-09-16 — セットアップと実測スループット（前段の検証ブランチ）

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp` アーキテクチャ）を RTX 3090 1枚 + ホストRAMに載せて
回すための、上流未マージPRを積んだ検証用ブランチです。**こちらは前段のスナップショット**で、
master追従と追加PRまで行ったのが [`speedup-2026-09-20`](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-20/speedup-bench/README.md) です。

- ブランチ: `speedup-2026-09-16`
- 起点: `ggml-org/llama.cpp` master `d1d3c3396`（2026-09-20 時点の親ブランチ基準）
- ブランチ上のコミット数: 35 / master比: 37 files, +2415/−286
- 次段: **[speedup-2026-09-20](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-20/speedup-bench/README.md)**（master再ベース＋Open PR追加、実測の詳細はこちら）

---

## 1. 検証機のセットアップ

検証機は次段ブランチと同じ1台です（スペックと混同しやすい点はそちらに集約）:

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB（`sm_86`, driver 615.71.09）|
| CPU | AMD Ryzen 9 7950X (16C/32T, AVX-512 VNNI) |
| RAM | DDR5 128 GB（4×32GB, 動作 4000 MT/s、モジュール定格は 5600 MT/s）|
| OS | Ubuntu 24.04 LTS (kernel 6.8) |
| CUDA | 13.4 |
| ターゲットモデル | `Qwen3.8-Flash-Next-Uncensored-IQ4_XS`（3分割, 92GB, QSA indexer 付き）|

> RAMが効く理由: 24GBのVRAMに収まらないMoE層はホストRAM側に置かれ、そのdecodeは
> **システムRAMの帯域**で律速されます（この機体は 4枚挿し=2DPC のため 4000 MT/s 動作）。

### ビルド

```bash
cd "$REPO"          # llama.cpp の作業コピー
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 28
```

### 実行時の共通条件

`-fa on --cache-type-k q8_0 --cache-type-v q8_0`、`--parallel 1`、モデル配置はllama.cppの自動fit
（VRAM 24GBに収まる範囲でレイヤーをオフロード、残りはホストRAM）。本番プリセットは `-c 262144`。

---

## 2. このブランチに入っているもの

上流未マージPRのみを積んだ状態です（Open PRの追加・master再ベースはしていません）。
PRはすべて `ggml-org/llama.cpp` のもので、**作者を併記**しています。

| PR | 作者 | 内容 | 09-16時点の扱い |
|---|---|---|---|
| [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) | [@JoursBleu](https://github.com/JoursBleu) | CUDA MMVQ: 行列幅に応じた動的 nwarps（MoE向け）| 取り込み |
| [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | [@BLSharda](https://github.com/BLSharda) | CUDA GDN chunked kernel の prefill 対応 | 取り込み |
| [#27044](https://github.com/ggml-org/llama.cpp/pull/27044) | [@glennneuber](https://github.com/glennneuber) | CUDA MMQ ids-path の tail padding を平坦化後の行数から算出 | 取り込み |
| [#28039](https://github.com/ggml-org/llama.cpp/pull/28039) | [@matteius](https://github.com/matteius) | 過大な norm グリッドを CUDA の 65,535 制限内に再配分 | 取り込み |
| [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) | [@coder543](https://github.com/coder543) | qwen4exp: lazy PLE table の direct read（prefill が2倍超）| 取り込み（次段で #29030 に置換）|
| [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) | [@abdel-darwish-27](https://github.com/abdel-darwish-27) | qwen4exp: QSA decode の gather ベース sparse attention | 取り込み |
| [#28432](https://github.com/ggml-org/llama.cpp/pull/28432) | [@am17an](https://github.com/am17an) | CUDA: top-k MoE を常に発火させる | 取り込み |
| [#28699](https://github.com/ggml-org/llama.cpp/pull/28699) | [@Rhonstin](https://github.com/Rhonstin) | qwen4exp: QSA indexer の incremental pooled-key cache | 取り込み |
| [#28770](https://github.com/ggml-org/llama.cpp/pull/28770) | [@am17an](https://github.com/am17an) | CUDA: qwen4 の sparse FA を有効化 | 取り込み（**上流にもマージ済み**）|
| [#28875](https://github.com/ggml-org/llama.cpp/pull/28875) | [@douyamv](https://github.com/douyamv) | CUDA: decode バッチでの tiny-N F32/F16 重みに mmvf を使用 | 取り込み（上流では未マージのままクローズ）|
| [#28389](https://github.com/ggml-org/llama.cpp/pull/28389) | [@TheArchitectit](https://github.com/TheArchitectit) | CUDA: in-place keys による CUB argsort 破損の修正 | 取り込み（**上流にもマージ済み**）|
| [#28901](https://github.com/ggml-org/llama.cpp/pull/28901) | [@am17an](https://github.com/am17an) | qwen4exp: hc ops の追加 | 取り込み（**上流にもマージ済み**）|
| [#28414](https://github.com/ggml-org/llama.cpp/pull/28414) | [@leshchukandrej](https://github.com/leshchukandrej) | `--prefetch-experts-slots`（MoE expert H2D 先読み）| **マージ後に revert**（後述）|

### QSA pooled cache（未コミットだった手元改修）

当時、稼働ビルドには **QSA pooled cache を sequence ごとの行に分離する改修**（per-sequence rows）と
調査用のログ計装が作業ツリーに未コミットで載っていました。後続で `local-qsa-ownrow-2026-09-20`
（ローカルブランチ）としてコミットされ、次段ブランチに取り込まれています。
**下記の実測値はこの未コミット改修込みのビルド**で測ったものです（ブランチ単体のビルドではありません）。

> 計装は `[TAG_QSA_DBG]` / `[TAG_QSA_GUARD]` / `[TAG_QSA_BIAS]` / `[TAG_QSA_WINDOW]` で、
> 環境変数（`LLAMA_QSA_GUARD`, `LLAMA_QSA_BIAS`, `LLAMA_QSA_WINDOW`）と
> 作業ディレクトリの `qsa-window.txt` で制御されます。

### 意図的に除外

- **#28414 `--prefetch-experts-slots`**: マージ後に revert しました。次段ブランチで再適用を試したところ
  **CUDA illegal memory access でクラッシュ**したため、除外の判断はそのまま維持しています。

---

## 3. 実測結果

測定方法: `llama-server` を起動し、`/v1/chat/completions` の応答 `timings` を採用
（`prompt_per_second` = pp、`predicted_per_second` = tg）。スクリプトは同梱の
[`bench.sh`](./bench.sh)。3列すべて **同じ機体・同じモデル（IQ4_XS, KV q8_0, `-fa on`,
`--load-mode none`）・同じプロンプト**で、バイナリだけを差し替えて測っています。

- 「適用前」= このブランチの分岐点 `d1d3c3396`（master）のビルド（2026-09-16 ビルド、2026-09-20 に再計測）
- 「このブランチ」= 下記PR群＋当時未コミットだったQSA per-sequence rows改修込みのビルド
- 「次段」= [speedup-2026-09-20](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-20/speedup-bench/README.md)（master再ベース＋Open PR追加）

| 条件 | 適用前 (master `d1d3c3396`) | このブランチ | 次段 (speedup-2026-09-20) |
|---|---|---|---|
| ctx 32768 / prompt≈2k | pp 254.1 t/s, tg 27.82 t/s | pp **373.8 (+47%)**, tg 25.65 (−7.8%) | pp **385.1 (+52%)**, tg 26.64 (−4.2%) |
| ctx 131072 / prompt≈32k | pp 275.6 t/s, tg 23.66 t/s | pp **418.8 (+52%)**, tg 23.69 (±0%) | pp **419.5 (+52%)**, tg 23.80 (+0.6%) |

**このブランチの主効果は prefill**です（長文脈で **+52%**、32kでも +47%）。長い文脈を読ませる
待ち時間がそのまま短くなります。一方 **decode は短い文脈でわずかに下がり（−8%）、長文脈では同等**
でした。QSAまわりのPR（gather型 sparse attention / sparse FA / pooled-key cache の再設計）が
索引計算を速くする代わりに、短prompt・単発decodeでは固定費が乗る形になっています。

長文脈（131k）で次段との差がほとんど出ないのは、次段で追加したPRの大半が
「ホストオフロード MoE のVRAMキャッシュ」「TOP_K の radix 化」など、
この24GB機のワークロードでは発火しない／逆効果になるものだったためです（詳細は次段のREADME）。
つまり **09-16 → 09-20 の実質的な利得は短〜中コンテキストの decode 数%**で、
**大きな段差は master → 09-16 の prefill**にあります。

---

## 4. 使い方

```bash
git checkout speedup-2026-09-16
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 28

MODEL_DEFAULT=/path/to/target.gguf ./speedup-bench/bench.sh \
    -b build/bin/llama-server -l old32k -c 32768 -r 200
```

`bench.sh` の主なオプション: `-b` バイナリ / `-l` ラベル / `-m` モデル / `-d` ドラフト /
`-n` `--spec-draft-n-max` / `-c` コンテキスト / `-r` プロンプト反復数 / `-t` max_tokens /
`-f` 追加フラグ / `-e` 環境変数。`-h` でヘルプ。

---

## 5. 謝辞

このブランチは上流コミュニティの成果の上に成り立っています。

- **llama.cpp / ggml-org のメンテナと全contributorの皆さん** — ベースとなるmaster、CUDAバックエンド、
  `qwen4exp` アーキテクチャの実装を提供してくれています。本リポジトリはMITライセンスの元でのforkです。
- **取り込んだPRの作者の方々**:
  [@JoursBleu](https://github.com/JoursBleu)（#20831）/
  [@BLSharda](https://github.com/BLSharda)（#26001）/
  [@glennneuber](https://github.com/glennneuber)（#27044）/
  [@matteius](https://github.com/matteius)（#28039）/
  [@coder543](https://github.com/coder543)（#28136）/
  [@abdel-darwish-27](https://github.com/abdel-darwish-27)（#28213）/
  [@am17an](https://github.com/am17an)（#28432, #28770, #28901）/
  [@Rhonstin](https://github.com/Rhonstin)（#28699）/
  [@douyamv](https://github.com/douyamv)（#28875）/
  [@TheArchitectit](https://github.com/TheArchitectit)（#28389）/
  [@leshchukandrej](https://github.com/leshchukandrej)（#28414）
- **モデルの公開元**: [Unsloth](https://huggingface.co/unsloth)（`UD-Q4_K_XL` と共有MTPドラフト）/
  orcarouter（`Qwen3.8-Flash-Next-Uncensored-IQ4_XS`）/ 元モデルの **Qwen** チーム。
- `bench.sh` と本README、および上記の実測は本fork側の作業です。PRの採用・除外の判断は
  **この機体のワークロードでの実測に基づくもので、上流や作者の判断とは独立**です。

> 取り込みは検証目的です。#28414 の除外や #28136 の置換のように、作者の想定と異なる扱いを
> している箇所があります。異論があれば該当PRの採用を止めます。