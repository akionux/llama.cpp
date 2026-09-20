# speedup-2026-09-20 — セットアップと実測スループット

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp` アーキテクチャ）を RTX 3090 1枚 + ホストRAMに載せて
回すための、上流未マージPRを積んだ検証用ブランチです。**どの構成でどれだけ速度が出るか**を
再現できるようにしてあります。

- ブランチ: `speedup-2026-09-20`
- 起点: `ggml-org/llama.cpp` master `f072b1037`（2026-09-20）
- master比: 59 files, +3885/−313
- 検証機: デスクトップ1台（スペックは下記）
- 前段ブランチ: [`speedup-2026-09-16`](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-16/speedup-bench/README.md)（本READMEの実測で「旧 build」側。セットアップと当時のPR構成はそちら）

---

## 1. 検証機のセットアップ

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB（`sm_86`, driver 615.71.09）|
| CPU | AMD Ryzen 9 7950X (16C/32T, AVX-512 VNNI) |
| RAM | DDR5 128 GB（4×32GB, **動作 4000 MT/s**、モジュール定格は 5600 MT/s）|
| OS | Ubuntu 24.04 LTS (kernel 6.8) |
| CUDA | 13.4 |
| ターゲットモデル | `Qwen3.8-Flash-Next-Uncensored-IQ4_XS`（3分割, 92GB, QSA indexer 付き）|
| 対照モデル | `Qwen3.8-Flash-Next-UD-Q4_K_XL`（unsloth, 4分割, 104GB）|
| MTPドラフト | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`（unsloth, `blk.48.nextn.*`）|

> RAMが効く理由: 24GBのVRAMに収まらないMoE層はホストRAM側に置かれ、そのdecodeは
> **システムRAMの帯域**で律速されます。この機体は 4枚挿し（2DPC）のため 4000 MT/s 動作で、
> 定格 5600 MT/s のモジュール性能を出していません。ホストオフロード主体のこの構成では、
> DRAM速度の引き上げ（もしくはメモリチャネルあたりの枚数削減）がそのまま decode 速度に効く余地があります。

### ビルド

```bash
cd "$REPO"          # llama.cpp の作業コピー
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 28
```

- `--generate-code=arch=compute_86,code=[sm_86]`（RTX 3090）
- `GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`, `GGML_CUDA_GRAPHS=ON`
- 実運用は systemd user service から `--models-preset ./config.ini` で起動

### 実行時の共通条件

`-fa on --cache-type-k q8_0 --cache-type-v q8_0`、`--parallel 1`、モデル配置はllama.cppの自動fit
（VRAM 24GBに収まる範囲でレイヤーをオフロード、残りはホストRAM）。本番プリセットは `-c 262144`。

---

## 2. このブランチに入っているもの

PRはすべて `ggml-org/llama.cpp` のもので、**作者を併記**しています。

### 2.1 上流未マージPR（`speedup-2026-09-16` から継承）

| PR | 作者 | 内容 |
|---|---|---|
| [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) | [@JoursBleu](https://github.com/JoursBleu) | CUDA MMVQ: 行列幅に応じた動的 nwarps（MoE向け）|
| [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | [@BLSharda](https://github.com/BLSharda) | CUDA GDN chunked kernel の prefill 対応 |
| [#27044](https://github.com/ggml-org/llama.cpp/pull/27044) | [@glennneuber](https://github.com/glennneuber) | CUDA MMQ ids-path の tail padding を平坦化後の行数から算出 |
| [#28039](https://github.com/ggml-org/llama.cpp/pull/28039) | [@matteius](https://github.com/matteius) | 過大な norm グリッドを CUDA の 65,535 制限内に再配分 |
| [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) | [@abdel-darwish-27](https://github.com/abdel-darwish-27) | qwen4exp: QSA decode の gather ベース sparse attention |
| [#28432](https://github.com/ggml-org/llama.cpp/pull/28432) | [@am17an](https://github.com/am17an) | CUDA top-k MoE を常に発火させる |
| [#28699](https://github.com/ggml-org/llama.cpp/pull/28699) | [@Rhonstin](https://github.com/Rhonstin) | qwen4exp: QSA indexer の incremental pooled-key cache |
| [#28770](https://github.com/ggml-org/llama.cpp/pull/28770) | [@am17an](https://github.com/am17an) | CUDA: qwen4 の sparse FA を有効化 — **上流にマージ済みのため master 側を採用** |
| [#28875](https://github.com/ggml-org/llama.cpp/pull/28875) | [@douyamv](https://github.com/douyamv) | CUDA: decode バッチでの tiny-N F32/F16 重みに mmvf を使用 |
| [#28389](https://github.com/ggml-org/llama.cpp/pull/28389) | [@TheArchitectit](https://github.com/TheArchitectit) | CUDA: in-place keys による CUB argsort 破損の修正 — **上流にマージ済みのため master 側を採用** |
| [#28901](https://github.com/ggml-org/llama.cpp/pull/28901) | [@am17an](https://github.com/am17an) | qwen4exp: hc ops の追加 — **上流にマージ済みのため master 側を採用** |
| [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) | [@coder543](https://github.com/coder543) | qwen4exp: lazy PLE table の direct read — 上流の #29030 が supersede のため revert |

### 2.2 今回追加（Open PR / 手元ブランチ）

| 追加 | 作者 | 内容 |
|---|---|---|
| [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) | [@danielhanchen](https://github.com/danielhanchen) | models: **Qwen3.8-Flash-Next MTP**（共有MTPモジュール。`--spec-type draft-mtp`）|
| [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) | [@rmonsurate](https://github.com/rmonsurate) | qwen4exp: NextN/MTP ドラフトヘッドの追加（上記MTPの構成要素）|
| [#28097](https://github.com/ggml-org/llama.cpp/pull/28097) | [@TheArchitectit](https://github.com/TheArchitectit) | qwen4exp: ドラフトヘッドのみのGGUF（unslothレイアウト）対応＋draft-logits 修正 |
| [#29030](https://github.com/ggml-org/llama.cpp/pull/29030) | [@pwilkin](https://github.com/pwilkin) | lazy tensor 行の direct read（OS非依存の再実装、#28136 置き換え）|
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) | [@csantiago78](https://github.com/csantiago78) | GPU常駐 MoE expert LRU cache（`--moe-expert-cache N`）|
| [#28498](https://github.com/ggml-org/llama.cpp/pull/28498) | [@eapache](https://github.com/eapache) | KV cache 回転メタデータの保存修正 |
| [#28671](https://github.com/ggml-org/llama.cpp/pull/28671) | [@Inovello](https://github.com/Inovello) | CUDA: 幅広行向けに CUB fallback の TOP_K を radix-select 化 |
| [#28713](https://github.com/ggml-org/llama.cpp/pull/28713) | [@praneshgo](https://github.com/praneshgo) | CUDA: 行数の多い TOP_K を radix 化（#28671 と統合: radixを常時コンパイル、しきい値 8192列＋行数ゲート）|
| [#29166](https://github.com/ggml-org/llama.cpp/pull/29166) | [@akionux](https://github.com/akionux) | unified cache 複数sequence時の blk_bias インデックス修正（本forkのPR。ローカル版を優先採用）|
| ローカル（OWNROW） | [@akionux](https://github.com/akionux) | QSA pooled cache を **sequenceごとの行**に分離（+ window/guard/bias の計装）|

> **注意**: ローカル OWNROW 分には `[TAG_QSA_DBG]` 等の調査用ログ計装が残っています。
> 計装は環境変数（`LLAMA_QSA_GUARD`, `LLAMA_QSA_BIAS`, `LLAMA_QSA_WINDOW`）と
> 作業ディレクトリの `qsa-window.txt` で制御され、既定で一部が有効です。
> 上流提出用の整理版は別途必要です。

### 2.3 意図的に除外

- **#28414 `--prefetch-experts-slots`**: この構成で **CUDA illegal memory access でクラッシュ**しました
  （`ggml_backend_cuda_synchronize` で illegal memory access）。再適用しない判断です。

---

## 3. 実測結果

測定方法: `llama-server` を起動し、`/v1/chat/completions` の応答 `timings` を採用
（`prompt_per_second` = pp、`predicted_per_second` = tg）。スクリプトは同梱の
[`bench.sh`](./bench.sh)。

```bash
MODEL_DEFAULT=<target.gguf> ./bench.sh -b <bin>/llama-server -l 32k -c 32768 -r 200
```

### 3.1 リベースの純効果（ターゲット = IQ4_XS, 本番相当）

| 条件 | 旧 build ([`speedup-2026-09-16`](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-16/speedup-bench/README.md)) | 新 build (`speedup-2026-09-20`) |
|---|---|---|
| ctx 32768 / prompt≈2k | pp 373.8 t/s, tg 25.65 t/s | **pp 385.1 (+3.0%), tg 26.64 (+3.9%)** |
| ctx 131072 / prompt≈32k | pp 418.8 t/s, tg 23.69 t/s | pp 419.5 (+0.2%), tg 23.80 (+0.5%) |
| 本番プリセット (ctx 262144) | — | 疎通OK, tg 21.2 t/s（ロード直後の単発値）|

> この表の増分は **前段ブランチ比**です。分岐前のmaster（`d1d3c3396`）と比べると **prefill は +52%**
> （32k: 254.1 → 385.1 t/s、131k: 275.6 → 419.5 t/s）、decode は短文脈で −4%・長文脈で同等で、
> **大きな段差は前段ブランチ側（master → 09-16）の prefill** にあります。
> 3世代の比較表は[前段ブランチのREADME](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-16/speedup-bench/README.md)参照。

### 3.2 追加PRの効き（すべて新 build）

| 構成 | 結果 |
|---|---|
| 131k ＋ `GGML_CUDA_TOPK_RADIX_MIN_ROWS=1`（#28713 を強制発火）| tg 23.43 t/s → **利得なし**（QSA indexer の TOP_K は4行で、しきい値8行未満。CUB の DeviceTopK が使えるビルドでは元の経路が既に速い）|
| 16k ＋ `--moe-expert-cache 8 --moe-expert-cache-inserts 4`（#27861）| tg 25.56 vs 26.55（cache無し）→ **−3.7%**（VRAMが既に 22.8/24.5GB）|
| 131k ＋ `--prefetch-experts-slots 2`（#28414）| **クラッシュ**（CUDA illegal memory access）|
| 16k, `-ngl 48 --n-cpu-moe 48` ＋ MTP(n-max 3)、ドラフト=unsloth版 | tg 20.91 vs 21.89（MTP無し）→ −4.5%、受け入れ率 47.9% |
| 16k, 同上 ＋ MTP(n-max 6) | tg 21.13 → −3.5%、受け入れ率 40.0% |

### 3.3 MTP（`--spec-type draft-mtp`）は「ドラフト一致」でのみ効く

unsloth `UD-Q4_K_XL` ＋ 同一配布元の共有MTPドラフト、ctx 16384、既定fit:

| 構成 | 結果 |
|---|---|
| MTP無し | tg **21.86** t/s |
| MTP (n-max 3) | tg **24.35** t/s（**+11.4%**）、受け入れ率 **63.5%**、mean len 2.88 |

同じMTPドラフトを **別系統のuncensoredモデル（IQ4_XS）** に付けると受け入れ率 40〜48% まで落ち、
速度は非MTPよりわずかに悪化しました。**MTPは同一チェックポイント由来のドラフトと組で使うこと**が条件です。

### 3.4 まとめ

- master追従＋新PRの純効果は **短〜中コンテキストで decode +3.9% / prefill +3.0%**、長コンテキストではほぼ横ばい。
- **MTP は一致ドラフトで +11%**。本番uncensoredモデルには nextn テンソルが無いため、そのままでは使えません
  （使うなら同モデル用のMTPヘッドを `convert_hf_to_gguf.py` で書き出す必要があります）。
- ホストオフロード向けの `--moe-expert-cache` / `--prefetch-experts-slots` は、この24GB機では
  **逆効果またはクラッシュ**。VRAMに余裕のある構成でのみ意味があります。
- QSA indexer の TOP_K radix化（#28671/#28713）は、行数が少ない現行ワークロードでは発火せず効果なし。
- ホストオフロード主体の構成なので、**DRAM帯域（現状 4000 MT/s 動作）が decode の律速**です。

---

## 4. 使い方

```bash
# リベース後のブランチでビルド
git checkout speedup-2026-09-20
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 28

# 速度測定（pp/tg を JSON timings から取得）
MODEL_DEFAULT=/path/to/target.gguf ./speedup-bench/bench.sh \
    -b build/bin/llama-server -l myrun -c 32768 -r 200

# MTP を試す（ターゲットと同一配布元のドラフトを使う）
MODEL_DEFAULT=/path/to/UD-Q4_K_XL-00001-of-00004.gguf ./speedup-bench/bench.sh \
    -b build/bin/llama-server -l mtp -c 16384 -r 100 -d /path/to/mtp-...-shared-Q8_0.gguf -n 3
```

`bench.sh` の主なオプション: `-b` バイナリ / `-l` ラベル / `-m` モデル / `-d` ドラフト /
`-n` `--spec-draft-n-max` / `-c` コンテキスト / `-r` プロンプト反復数 / `-t` max_tokens /
`-f` 追加フラグ / `-e` 環境変数。`-h` でヘルプ。

---

## 5. 謝辞

このブランチは上流コミュニティの成果の上に成り立っています。

- **llama.cpp / ggml-org のメンテナと全contributorの皆さん** — ベースとなるmaster、CUDAバックエンド、
  `qwen4exp` アーキテクチャの実装、および本ブランチが積んでいるPRのレビューを担っています。
  本リポジトリはMITライセンスの元でのforkです。
- **取り込んだPRの作者の方々**:
  [@JoursBleu](https://github.com/JoursBleu)（#20831）/
  [@BLSharda](https://github.com/BLSharda)（#26001）/
  [@glennneuber](https://github.com/glennneuber)（#27044）/
  [@matteius](https://github.com/matteius)（#28039）/
  [@abdel-darwish-27](https://github.com/abdel-darwish-27)（#28213）/
  [@am17an](https://github.com/am17an)（#28432, #28770, #28901）/
  [@Rhonstin](https://github.com/Rhonstin)（#28699）/
  [@douyamv](https://github.com/douyamv)（#28875）/
  [@TheArchitectit](https://github.com/TheArchitectit)（#28389, #28097）/
  [@coder543](https://github.com/coder543)（#28136）/
  [@pwilkin](https://github.com/pwilkin)（#29030）/
  [@danielhanchen](https://github.com/danielhanchen)（#28243）/
  [@rmonsurate](https://github.com/rmonsurate)（#27836）/
  [@csantiago78](https://github.com/csantiago78)（#27861）/
  [@eapache](https://github.com/eapache)（#28498）/
  [@Inovello](https://github.com/Inovello)（#28671）/
  [@praneshgo](https://github.com/praneshgo)（#28713）/
  [@leshchukandrej](https://github.com/leshchukandrej)（#28414, 不採用）
- **MTPまわり**: #28243/#27836/#28097 の作者（[@danielhanchen](https://github.com/danielhanchen),
  [@rmonsurate](https://github.com/rmonsurate), [@TheArchitectit](https://github.com/TheArchitectit)) —
  ドラフト一致時の +11% はこの3件の上に成り立っています。
- **モデルとドラフトの公開元**: [Unsloth](https://huggingface.co/unsloth)（`UD-Q4_K_XL` と
  共有MTPドラフト）/ orcarouter（`Qwen3.8-Flash-Next-Uncensored-IQ4_XS`）/
  元モデルの **Qwen** チーム。
- `bench.sh` と本README、実測、QSA OWNROW 改修、#29166 は本fork側の作業です。PRの採用・除外の
  判断は **この機体のワークロードでの実測に基づくもので、上流や作者の判断とは独立**です。

> 取り込みは検証目的です。#28414 の不採用、#28136 の置換のように、作者の想定と異なる扱いを
> している箇所があります。異論があれば該当PRの採用を止めます。