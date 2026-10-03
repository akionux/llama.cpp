# speedup-2026-10-03 — 上流master追随と実測スループット

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp` アーキテクチャ）を RTX 3090 1枚 + ホストRAMに載せて
回すための検証用ブランチです。**どの構成でどれだけ速度が出るか**を再現できるようにしてあります。

- ブランチ: `speedup-2026-10-03`
- 起点: `ggml-org/llama.cpp` master `bed0a8566`（2026-10-03）
- master比: 9 files, +1071/−44
- 前段ブランチ: [`speedup-2026-09-20`](https://github.com/akionux/llama.cpp/blob/speedup-2026-09-20/speedup-bench/README.md)

> **前段からの大きな変更**: 上流masterがこの2週間で、前段ブランチが積んでいた主要PR
> （QSA sparse attention、sparse FA、top-k MoE、hc ops、MTP、lazy read など）を**独自実装で取り込んだ**ため、
> 本ブランチは**未マージのまま残っているPRだけを積むスリム構成**に切り替えました。
> 詳しくは §2 を参照してください。

---

## 1. 検証機のセットアップ

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB（`sm_86`, driver 615.71.09）|
| CPU | AMD Ryzen 9 7950X (16C/32T, AVX-512 VNNI) |
| RAM | DDR5 128 GB（4×32GB, **動作 4000 MT/s**、モジュール定格は 5600 MT/s）|
| OS | Ubuntu 24.04.5 LTS (kernel 6.8.0) |
| CUDA | 13.4 |
| ターゲットモデル | `Qwen3.8-Flash-Next-Uncensored-Q4_K_M`（3分割, 119GB, QSA indexer 付き）|
| 対照クオンタイズ | `Qwen3.8-Flash-Next-Uncensored-IQ4_XS`（3分割, 92GB）|

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

### 2.1 積んでいるPR（上流に未マージのものだけ）

| PR | 作者 | 状態 | 内容 |
|---|---|---|---|
| [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) | [@JoursBleu](https://github.com/JoursBleu) | open | CUDA MMVQ: 行列幅に応じた動的 nwarps |
| [#28875](https://github.com/ggml-org/llama.cpp/pull/28875) | [@douyamv](https://github.com/douyamv) | closed (未merge) | CUDA: decode バッチでの tiny-N F32/F16 重みに mmvf を使用 |
| [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | [@BLSharda](https://github.com/BLSharda) | open | CUDA: GDN chunked kernel の prefill 対応 |

> 注意: #20831 の動的 nwarps 縮小は **RDNA3/RDNA4（AMD）専用**の分岐で、本機（NVIDIA SM86）では発火しません。
> #26001 の chunked GDN も `qwen4exp` の attention 経路ではなく GDN 層向けです。
> したがって §3 の速度は横ばいが想定どおりで、この2件は「将来のAMD/GDN構成への備え」として積んであります。

### 2.2 上流に取り込まれたため、この世代では落としたもの

前段ブランチが union として積んでいた以下のPRは、master 側で既に（多くは別実装として）実現されています。
本ブランチは master 側を採用し、PR側の適用は行っていません。

| PR | 作者 | master 側の状態 |
|---|---|---|
| [#28770](https://github.com/ggml-org/llama.cpp/pull/28770) | [@am17an](https://github.com/am17an) | **merged**（2026-09-20）— qwen4 の sparse FA |
| [#28432](https://github.com/ggml-org/llama.cpp/pull/28432) | [@am17an](https://github.com/am17an) | **merged**（2026-09-23）— top-k MoE を常に発火 |
| [#28901](https://github.com/ggml-org/llama.cpp/pull/28901) | [@am17an](https://github.com/am17an) | **merged**（2026-09-16）— qwen4exp hc ops |
| [#28699](https://github.com/ggml-org/llama.cpp/pull/28699) | [@Rhonstin](https://github.com/Rhonstin) | 未マージのまま。master は `kpool` 設計（`kpool_layout` / `kpool_access` / 系列ごとの `stale_pos`）で同領域を実装済み |
| [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) | [@abdel-darwish-27](https://github.com/abdel-darwish-27) | closed（未merge）。master は選択ベース（`build_qsa_sel`/`n_sel`）＋ sparse FA で実装 |
| [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) | [@danielhanchen](https://github.com/danielhanchen) | closed（未merge）。master は独自の MTP 実装（`graph_mtp`）を持つ |
| [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) / [#29030](https://github.com/ggml-org/llama.cpp/pull/29030) | [@coder543](https://github.com/coder543) / [@pwilkin](https://github.com/pwilkin) | master は `can_prefetch` / `llama_prefetch_rows` を持つ |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) | [@csantiago78](https://github.com/csantiago78) | 前段で実測 **−3.7%** だったため不採用 |
| [#28671](https://github.com/ggml-org/llama.cpp/pull/28671) / [#28713](https://github.com/ggml-org/llama.cpp/pull/28713) | [@Inovello](https://github.com/Inovello) / [@praneshgo](https://github.com/praneshgo) | 前段で実測 **利得なし** だったため不採用 |
| ローカル QSA 計装（OWNROW ほか） | [@akionux](https://github.com/akionux) | master の `kpool` は系列ごとの `stale_pos` を持ち、同種の問題を設計として扱う。旧API（`set_input_qsa`/`pooled_k`）は master から消えており、そのままでは適用できない |

### 2.3 意図的に除外（継続）

- **#28414 `--prefetch-experts-slots`**: この構成で **CUDA illegal memory access でクラッシュ**しました。
  再適用しない判断です。

---

## 3. 実測結果

測定方法: `llama-server` を起動し、`/v1/chat/completions` の応答 `timings` を採用
（`prompt_per_second` = pp、`predicted_per_second` = tg）。スクリプトは同梱の
[`bench.sh`](./bench.sh)。各条件を **2パス** 実行し、2パス目（ページキャッシュが温まった側）の値を採用しています。

```bash
MODEL_DEFAULT=<target.gguf> ./bench.sh -b <bin>/llama-server -l 32k -c 32768 -r 200
```

### 3.1 3ビルドの比較（ターゲット = Q4_K_M, 本番と同モデル）

| 条件 | 旧 union (`speedup-2026-09-20`) | master素 (PRなし) | 新 (`speedup-2026-10-03`) |
|---|---|---|---|
| ctx 16384 / prompt≈2k | pp 206.9, **tg 24.87** | pp 204.6, tg 24.61 | pp **208.8**, tg 24.82 |
| ctx 131072 / prompt≈32k | pp 220.5, **tg 22.41** | pp 216.8, tg 22.22 | pp **220.9**, tg 22.33 |

（pp/tg = tokens/s。master素は本ブランチの起点 `bed0a8566` そのものを別worktreeでビルドしたもの）

### 3.2 まとめ

- **リベースの速度影響は誤差範囲**（新 vs 旧union: prefill +0.9% / +0.2%、decode −0.2% / −0.3%）。
- 追加した3PRのうち2PR（#20831, #26001）は本機のワークロードで発火しないため、横ばいは想定どおり。
- **より重要な発見**: master素（PRなし）が長文脈で 旧union とほぼ同等（131k tg 22.22 vs 22.41 = −0.8%）。
  前世代で union が clean master に対して持っていた優位（当時のREADMEで 129.5k decode +53%）は、
  **上流masterが同等の attention 経路実装を取り込んだことで解消済み**です。
  つまりこの世代の「リベース」は速度のための作業ではなく、**機能追従と上流互換の維持**のためのものです。
- ホストオフロード主体の構成なので、依然 **DRAM帯域（現状 4000 MT/s 動作）が decode の律速**です。

---

## 4. 使い方

```bash
# このブランチでビルド
git checkout speedup-2026-10-03
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 28

# 速度測定（pp/tg を JSON timings から取得）
MODEL_DEFAULT=/path/to/target.gguf ./speedup-bench/bench.sh \
    -b build/bin/llama-server -l myrun -c 32768 -r 200
```

`bench.sh` の主なオプション: `-b` バイナリ / `-l` ラベル / `-m` モデル / `-d` ドラフト /
`-n` `--spec-draft-n-max` / `-c` コンテキスト / `-r` プロンプト反復数 / `-t` max_tokens /
`-f` 追加フラグ / `-e` 環境変数。`-h` でヘルプ。

---

## 5. 謝辞

このブランチは上流コミュニティの成果の上に成り立っています。

- **llama.cpp / ggml-org のメンテナと全contributorの皆さん** — ベースとなるmaster、CUDAバックエンド、
  `qwen4exp` アーキテクチャ、QSA、MTP まわりの実装を担っています。本リポジトリはMITライセンスの
  元でのforkです。
- **本ブランチが積んでいるPRの作者**: [@JoursBleu](https://github.com/JoursBleu)（#20831）/
  [@douyamv](https://github.com/douyamv)（#28875）/
  [@BLSharda](https://github.com/BLSharda)（#26001）
- **上流で取り込み済みとなったPRの作者**: [@am17an](https://github.com/am17an)（#28770, #28432, #28901）/
  [@Rhonstin](https://github.com/Rhonstin)（#28699）/
  [@abdel-darwish-27](https://github.com/abdel-darwish-27)（#28213）/
  [@danielhanchen](https://github.com/danielhanchen)（#28243）/
  [@coder543](https://github.com/coder543)（#28136）/
  [@pwilkin](https://github.com/pwilkin)（#29030）
- **モデルの公開元**: orcarouter（`Qwen3.8-Flash-Next-Uncensored`）/ 元モデルの **Qwen** チーム。
- `bench.sh` と本README、実測は本fork側の作業です。PRの採用・除外の判断は **この機体の
  ワークロードでの実測に基づくもので、上流や作者の判断とは独立**です。

> 取り込みは検証目的です。#27861/#28671/#28713 の不採用、#28213/#28243 をPRとして積まず
> master の実装を使う判断など、作者の想定と異なる扱いをしている箇所があります。
> 異論があれば該当PRの採用方針を変えます。
