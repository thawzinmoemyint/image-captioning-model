# VisionGPT-2 Image Captioning (improved)

An image captioning model that pairs a **Vision Transformer (ViT)** encoder with a **GPT-2** decoder. Cross-attention layers added to GPT-2 let the decoder read the image features. The model is trained and evaluated on **COCO 2017**.

This project builds on the public Kaggle notebook [VisionGPT2 Image Captioning (PyTorch)](https://www.kaggle.com/code/shreydan/visiongpt2-image-captioning-pytorch) by shreydan. The model architecture and training schedule come from that notebook. This repository fixes several correctness problems, adds faster decoding and beam search, and adds standard captioning metrics. Each change is measured against the original model.

Notebook: [`visiongpt-2-image-captioning.ipynb`](visiongpt-2-image-captioning.ipynb)

## Results

All scores are on the full COCO `val2017` split (5,000 images, ~25k reference captions). Both models were trained with the same budget: 150k training captions and 5 epochs.

| Model | Decoding | BLEU-1 | BLEU-4 | CIDEr |
|---|---|---|---|---|
| Original checkpoint (E0) | greedy | 0.661 | 0.235 | 0.787 |
| Original checkpoint (E1) | beam (k=3) | 0.647 | 0.245 | 0.811 |
| **This version (E3)** | greedy | 0.728 | 0.310 | 1.022 |
| **This version (E3)** | **beam (k=3)** | **0.728** | **0.323** | **1.058** |

With the same training budget, CIDEr goes from **0.81 to 1.06** (about +30%) and BLEU-4 goes from **0.245 to 0.323**.

**Decoding speed** (greedy, one image at a time, fp32, 20 tokens max):

| Device | Original (re-runs the full model per token) | Cached (this version) | Speedup | Same captions |
|---|---|---|---|---|
| GPU (Tesla T4) | 263 ms | 98 ms | **2.7x** | 100% |
| CPU | 3,361 ms | 624 ms | **5.4x** | 100% |

Training progress for E3 (per-epoch CIDEr is computed on 1,000 val images):

| Epoch | Train loss | Val loss | Val BLEU-4 | Val CIDEr |
|---|---|---|---|---|
| 0 | 4.868 | 3.101 | 0.168 | 0.558 |
| 1 | 2.570 | 2.283 | 0.236 | 0.822 |
| 2 | 2.262 | 2.161 | 0.261 | 0.890 |
| 3 | 2.003 | 2.077 | 0.300 | 1.036 |
| 4 | 1.832 | 2.071 | 0.312 | 1.075 |

> Validation perplexity here (7.9) is **not comparable** with the original notebook's ~8.0. The validation images are now unseen by the model, and the loss also includes predicting the first word. Compare runs on CIDEr and BLEU-4.

## What changed and why

| # | Change | Why it matters |
|---|---|---|
| 1 | **Validate on the official COCO `val2017` split** instead of a random split of caption rows | COCO has about 5 captions per image, so splitting by caption put the same images in both train and validation. The original validation perplexity was therefore too optimistic. |
| 2 | **Fixed the train/generation mismatch for the first word.** Training sequences are now `[BOS] caption [EOS]` | The original trained on `caption [EOS]` but started generation from `[BOS]`. The model never learned to predict the first word, and every token sat one position away from where it was trained. |
| 3 | **Encoder caching and a KV cache during generation** | The original re-ran all 12 ViT blocks and the whole decoder for every generated token. Now the image is encoded once and past keys and values are reused. The captions are identical and decoding is 2.7–5.4x faster. |
| 4 | **Beam search** and batched greedy decoding | The original only had sampling or greedy decoding, one image at a time. |
| 5 | **BLEU-1/4 and CIDEr evaluation** with `pycocoevalcap` | Perplexity alone doesn't measure caption quality. CIDEr is the standard captioning metric. |
| 6 | **Best checkpoint chosen by validation CIDEr.** A checkpoint is saved every epoch, training can resume, and a time guard stops cleanly before the Kaggle limit | The epoch with the best perplexity is not always the one with the best captions. Long runs can survive a session timeout. |
| 7 | Smaller fixes: `autocast` wraps only the forward pass, horizontal-flip augmentation is removed, and the ignored `always_apply` argument is removed | Flipping an image contradicts captions that say "left" or "right". The other two are correctness and cleanup fixes. |

Module and parameter names are unchanged, so **a checkpoint from the original notebook loads directly** into this model.

## Model

- **Encoder:** `vit_base_patch16_224` from `timm` (pretrained), 12 blocks, 224×224 input.
- **Decoder:** GPT-2 small (12 layers, 768-dim, 12 heads) initialised from Hugging Face weights, with a **cross-attention layer added to each block**. Decoder block *i* attends to the output of ViT block *i*.
- **Encode/decode split:** the ViT never reads the decoder, so running all ViT blocks first gives exactly the same result as the original interleaved loop. Because of this, generation can encode an image once and reuse it.
- **Attention:** uses `F.scaled_dot_product_attention`, which is mathematically the same as the original attention and uses fused kernels on GPU.
- **Size:** about 239M parameters.

## Training

- **Data:** COCO `train2017` captions. In `compare` mode, 150k captions (90,924 images) are sampled.
- **Augmentations** (`albumentations`): brightness/contrast, color jitter, shift-scale-rotate, hue/saturation. No horizontal flip.
- **Batching:** a custom `collate_fn` with dynamic padding. Padding is marked by length, because GPT-2 uses the same token for BOS, EOS and padding.
- **Optimizer:** Adam with OneCycleLR (max LR 1e-4), batch size 32, mixed precision (fp16).
- **Staged unfreezing:**
  - Epoch 0: only the cross-attention layers train.
  - Epoch 1: GPT-2 layers are unfrozen as well.
  - Epoch 2 onward: the whole model trains, including the ViT.
- **Time:** about 40 minutes per epoch on a Tesla T4.

## Caption generation

| Function | What it does |
|---|---|
| `generate` | Batched greedy decoding, or sampling with a `temperature`. Encodes the image once and uses the KV cache. |
| `generate_beam` | Batched beam search with the same caching. Defaults: `beam_size=3`, `length_penalty=1.0`. |
| `generate_naive` | The original algorithm, kept only as the speed baseline. |

## Experiments

Every result is appended to `results.csv`.

| ID | What | Needs training? |
|---|---|---|
| E0 | Original checkpoint, greedy, scored on `val2017` | No (needs the original `captioner.pt`) |
| E1 | Original checkpoint with beam search | No |
| E2 | Original checkpoint: naive vs cached decoding speed on GPU and CPU | No |
| E3 | Retrain with the fixes, `MODE='compare'` (same budget as the original: 150k captions, 5 epochs) | Yes |
| E4 | Final model, `MODE='full'` (all ~590k train captions, 8 epochs) | Yes |

Keep the budget fixed (`compare`) when comparing changes. Otherwise you can't tell whether a gain came from the method or from the extra training.

## How to run (Kaggle)

1. Create a Kaggle notebook from [`visiongpt-2-image-captioning.ipynb`](visiongpt-2-image-captioning.ipynb).
2. Settings: **GPU accelerator** and **Internet ON** (needed for `pip install` and the pretrained weights).
3. Add the input dataset **`awsaf49/coco-2017-dataset`**.
4. Optional, for E0–E2: add the original notebook's `captioner.pt` as an input and set `ORIGINAL_CKPT`. Skip that section if you don't have it.
5. In the configuration cell, set `MODE` to `'compare'` or `'full'` and run all cells.

Outputs go to `/kaggle/working/run_<EXP_NAME>/`:

| File | Contents |
|---|---|
| `best.pt` | Best checkpoint by validation CIDEr |
| `last.pt` | Latest checkpoint, used for resuming |
| `val2017_beam_predictions.json` | Beam-search captions for all val images |
| `captioner_fp16.pt` | fp16 export for deployment |

To continue an interrupted run in a new session, set `cfg.resume_from` to a previous `last.pt`.

## Using the exported model

`captioner_fp16.pt` holds fp16 weights without the unused attention-mask buffers (555 MB). Load it with `strict=False`:

```python
model = VisionGPT2Model(model_config)
missing, unexpected = model.load_state_dict(
    torch.load('captioner_fp16.pt', map_location='cpu'), strict=False)
assert all(k.endswith('.attn.mask') for k in missing) and not unexpected
model = model.float().eval()
```

Then caption images with `generate` or `generate_beam` from the notebook.

## Notes on evaluation

Captions are lower-cased and stripped of punctuation before scoring, instead of using the official PTB tokenizer, which requires Java. Scores are consistent across every run in this project but may differ slightly from numbers reported in papers.

## Credits

- Original notebook and architecture: [shreydan, *VisionGPT2 Image Captioning (PyTorch)*](https://www.kaggle.com/code/shreydan/visiongpt2-image-captioning-pytorch)
- Pretrained models: [`timm`](https://github.com/huggingface/pytorch-image-models) ViT-B/16 and [Hugging Face](https://huggingface.co/gpt2) GPT-2
- Dataset: [COCO 2017](https://cocodataset.org/)
- Metrics: [`pycocoevalcap`](https://github.com/salaniz/pycocoevalcap)
