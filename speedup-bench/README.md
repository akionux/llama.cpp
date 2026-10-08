# speedup-2026-10-08 — 上流master追随リベース + MoE専門家GPUキャッシュの採用

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp` アーキテクチャ）を RTX 3090 1枚 + ホストRAMに載せて
回すための検証用ブランチです。**どの構成でどれだけ速度が出るか**を再現できるようにしてあります。

- ブランチ: `speedup-2026-10-08`
- 起点: `ggml-org/llama.cpp` master `06cad0b9e`（2026-10-07）
- master比: 10 files, +1299/−11
- 前段ブランチ: `speedup-2026-10-03`

> **前段からの要点**: 上流masterに **#29887 `--moe-cache-mib`（MoE専門家のGPU LRUキャッシュ）が
> merge 済み**（`d6cf9acb2`）でした。本機で実測したところ、条件付きで **decode +19〜50%** の効果があり、
> **本番プリセット（ctx 262144）に `moe-cache-mib = 8192` として採用**しました（§4）。

---

## 1. 検証機のセットアップ

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB（`sm_86`）|
| CPU | AMD Ryzen 9 7950X (16C/32T, AVX-512 VNNI) |
| RAM | DDR5 128 GB（4×32GB, **動作 4000 MT/s**、モジュール定格は 5600 MT/s）|
| OS | Ubuntu 24.04.5 LTS |
| CUDA | 13.4 |
| ターゲットモデル | `Qwen3.8-Flash-Next-Uncensored-Q4_K_M`（3分割, 119GB, QSA indexer 付き）|

> RAMが効く理由: 24GBのVRAMに収まらないMoE層はホストRAM側に置かれ、そのdecodeは
> **システムRAMの帯域**で律速されます。本機は 4枚挿し（2DPC）のため 4000 MT/s 動作です。
> なお本モデル（119GB）はRAM 124GiBに対して大きく、**完全常駐は不可能**なので、測定は必ず
> ページキャッシュが温まったパス（2パス目）を採用してください。

### ビルド

```bash
cd "$REPO"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DGGML_NATIVE=ON -DGGML_CUDA_GRAPHS=ON \
      -DGGML_CUDA_FA_QUANTS='q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16'
cmake --build build --config Release -j 28
```

実運用は systemd user service から `--models-preset ./config.ini` で起動。

---

## 2. この世代の変更

### 2.1 積んでいるPR（上流に未マージのものだけ）

| PR | 作者 | 状態 | 内容 | 本機での効果 |
|---|---|---|---|---|
| [#28875](https://github.com/ggml-org/llama.cpp/pull/28875) | [@douyamv](https://github.com/douyamv) | closed (未merge) | CUDA: decode バッチの tiny-N F32/F16 重みに mmvf | ほぼ無し |
| [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | [@BLSharda](https://github.com/BLSharda) | open | CUDA: GDN chunked kernel の prefill 対応 | qwen4exp の attention 経路ではないため無し |

### 2.2 この世代で落としたPR

- **#20831（動的 nwarps）**: 上流の [#29184](https://github.com/ggml-org/llama.cpp/pull/29184)
  「shared experts を MMVQ に融合」が `calc_launch_params` のシグネチャを変えたため、リベースで
  `mmvq.cu` が衝突。RDNA3/RDNA4 専用分岐で本機（SM86）では発火しないので、上流側を採用して落としました。

### 2.3 上流に merge 済み（本ブランチの目玉）

| PR | 作者 | 内容 |
|---|---|---|
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) | [@am17an](https://github.com/am17an) | **`--moe-cache-mib N`**: ホストRAM上の専門家をGPUのLRUキャッシュに載せ、MUL_MAT_ID をGPUで実行（ミス分のみPCIe転送。バッチ≤32トークンのみ）|

---

## 3. 実測 1: リベース自体の影響（中立）

測定方法: `llama-server` の `/v1/chat/completions` レスポンス `timings`。
プロンプトは同梱 [`bench.sh`](./bench.sh) の定型文（`-b 4096 -ub 2048 --parallel 1`、各条件2パス、2パス目採用）。

| 条件 | 本番 bin (`speedup-2026-10-03`) | リベース後 (`speedup-2026-10-08`) | 差 |
|---|---|---|---|
| ctx 32768 / prompt 30k tok | pp 550.9 / tg 23.16 | pp 542.4 / **tg 23.52** | tg +1.6% |
| ctx 16384 / prompt 2.5k tok | pp 357.6 / tg 25.15 | pp 357.2 / **tg 25.33** | tg +0.7% |

上流が前段で積んでいたPR群を独自実装で取り込み済みのため、**リベースは速度のための作業ではなく
機能追従のための作業**という前段の結論を再確認しました。

---

## 4. 実測 2: MoE専門家GPUキャッシュ `--moe-cache-mib`（#29887）

### 4.1 前提条件: `--load-mode none` でなければ逆効果

同じキャッシュ設定でも、**mmap有効（llama.cppの既定）だと decode が悪化**します。

| cache MiB | pp t/s | tg t/s（mmap有効）| tg t/s（`--load-mode none`）|
|---|---|---|---|
| 0 | 542 | 23.5 | 22.3 |
| 4096 | 550 | 17.6（**−25%**）| – |
| 8192 | 540 | 20.8（−12%）| 27.3（**+23%**）|

理由: キャッシュミス時のアップロード元が mmap の pageable メモリになり、ページフォルト/
ステージングを毎回払うためです。**本番プリセットは `load-mode = none` なので後者の列が該当**します。

### 4.2 ctx 32768 — 膝と崖（`--load-mode none`, prompt 30,075 tok）

| cache MiB | pp t/s | tg t/s | 対基準 | ヒット率 | CUDA0 model buffer |
|---|---|---|---|---|---|
| 0 | 691.9 | 22.26 | – | – | 20409 MiB |
| 6144 | 690.0 | 25.67 | +15% | 59.5% | 13853 |
| 8192 | 684.5 | 27.32 | +23% | 64.4% | 11652 |
| 12288 | 679.3 | 30.38 | +37% | 73.3% | 7308 |
| **13312** | 676.3 | **32.35 / 33.41** | **+45〜50%** | – | – |
| 14336 | 677.3 | 31.94 | +43% | 76.4% | 5558 |
| 16384 | 663.5 | 19.10 | **−14%** | 86.1% | 3724 |
| 18432 / 20480 | 635 | 9.8 / 9.9 | **−56%** | 89% | 2515 |

### 4.3 本番 ctx 262144（KV q8_0 が約3GiB食うぶん崖が手前に来る）

| cache MiB | pp t/s | tg t/s | 対基準 |
|---|---|---|---|
| 0 | 677.7 / 678.2 | 20.93 / 20.88 | – |
| 4096 | 683.1 | 21.78 | +4% |
| 6144 | 678.9 | 23.45 | +12% |
| **8192** | 674.7 | **24.90** | **+19%** |
| 10240 | 660.3 | 19.35 | −7.5% |
| 12288 | 637.7 | 13.17 | −37% |
| 14336 | 619.6 | 9.53 | −54% |

### 4.4 崖の法則（重要）

- 最適点のすぐ上で decode が **1/3〜1/2 に崩壊**します。崩壊時は `--fit` が専門家層を
  すべてホストへ押し出し、`CUDA0 model buffer size` が 2.5〜3.7 GiB（＝attentionのみ）まで落ちます。
- 崖の位置は**空きVRAMに比例**するため、**最適値はctxごとに実測が必要**です:
  ctx 32768 → 最適 13312 / 崩壊 16384、ctx 262144 → 最適 8192 / 崩壊 10240。
- **`--fit` はキャッシュを配置計算に含めます**（model bufferが自動的に縮む＝丸ごと層をキャッシュに
  譲る）。`-ncmoe`/`-ngl` の手動調整は不要でした。
- 参考ログ（`-lv 4` が必要、本番ctx 262144・8192設定時の実測例）:
  `impl: CUDA0 MoE cache size = 8188.18 MiB for 73300.00 MiB of host experts`
  → `llama_moe_cache: ubatch <= 8: hits = 57902, misses = 36214, hit rate = 61.52%, uploaded = 112503.12 MiB`
- 再現性: 同一条件の繰り返しで ±1.0%（13312: 33.41 → 32.35）。

---

## 5. 本番への適用（2026-10-08）

1. `build/bin` を `speedup-2026-10-08` のビルドに差し替え
   （ロールバック用スナップショット = `build/bin.2026-10-03`）。
   `--moe-cache-mib` は旧ビルド（`306ae5f6a`）のプリセットでは
   `option 'moe-cache-mib' not recognized in preset` で起動に失敗するため、ビルド差し替えは必須。
2. `config.ini` の `[qwen3.8-flash-local]` に `moe-cache-mib = 8192` を追加
   （バックアップ = `config.ini.bak.2026-10-08`）。
3. systemd user service を restart → `/v1/models` の `status.args` で
   `--moe-cache-mib 8192` / `--load-mode none` / `--cache-type-k,v q8_0` を確認。

### ルーター経由の before / after（ctx 262144、prompt 30,061 tok、ユニークプロンプト×2）

| | pp t/s | tg t/s |
|---|---|---|
| before | 683.7 / 680.0 | 20.02 / 20.30 |
| after | 688.9 / 688.7 | **26.28 / 25.96** |

→ **decode +30% / prefill は横ばい**（`-lv 4` 付きスタンドアロンA/Bでは +19%）。
表示トークンあたりで見ると、200トークン生成が **10.0s → 7.7s**。

---

## 6. 使い方

```bash
git checkout speedup-2026-10-08
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j 28

# キャッシュの膝を探す（ctxは本番と同じ値で）
MODEL_DEFAULT=<target.gguf> ./speedup-bench/bench.sh \
    -b build/bin/llama-server -l myrun -c 262144 -r 3000 -t 200 \
    -f "-b 4096 -ub 2048 --parallel 1 --load-mode none -lv 4 --moe-cache-mib 8192"
```

`bench.sh` の主なオプション: `-b` バイナリ / `-l` ラベル / `-m` モデル / `-d` ドラフト /
`-n` `--spec-draft-n-max` / `-c` コンテキスト / `-r` プロンプト反復数 / `-t` max_tokens /
`-f` 追加フラグ / `-e` 環境変数。`-h` でヘルプ。

---

## 7. 謝辞

- **llama.cpp / ggml-org のメンテナと全contributorの皆さん** — ベースとなるmaster、CUDAバックエンド、
  `qwen4exp` アーキテクチャ、QSA、MTP、そして本世代の主役である **#29887 MoEキャッシュ** の実装を
  担っています。本リポジトリはMITライセンスの元でのforkです。
- **本ブランチが積んでいるPRの作者**: [@douyamv](https://github.com/douyamv)（#28875）/
  [@BLSharda](https://github.com/BLSharda)（#26001）
- **上流に取り込み済みとなったPRの作者**: [@am17an](https://github.com/am17an)（#29887, ほか多数）
- **モデルの公開元**: orcarouter（`Qwen3.8-Flash-Next-Uncensored`）/ 元モデルの **Qwen** チーム。
- `bench.sh` と本README、実測は本fork側の作業です。PRの採用・除外の判断は **この機体の
  ワークロードでの実測に基づくもので、上流や作者の判断とは独立**です。
