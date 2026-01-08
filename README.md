
## VAD pipelines

### 1) Scan dataset once (create manifest)

Creates a lightweight manifest (`filelist.txt`) of absolute audio paths.

```bash
python scan.py /path/to/dataset
```

This writes to `manifests/<dataset>.txt` by default.

### 2) Run multi-GPU pyannote VAD (torchrun)

`vad2.py` shards the manifest across ranks (one process per GPU) and writes per-file JSONL.

Best practice for Hugging Face tokens: set them via environment variables (don’t hardcode or commit).

```bash
export HF_TOKEN="<your_hf_token>"
torchrun --standalone --nproc_per_node=4 vad2.py metadata/<dataset>_<DDMMYY>/filelist.txt
```

Outputs (default):
- `metadata/<dataset>_<DDMMYY>/metadata.rank*.jsonl`
- `metadata/<dataset>_<DDMMYY>/global.jsonl`
- `metadata/<dataset>_<DDMMYY>/durations/{speech_durations,nospch_durations}.rank*.npy`

If you prefer a single shared JSONL (slower due to locking):

```bash
torchrun --standalone --nproc_per_node=4 vad2.py metadata/<dataset>_<DDMMYY>/filelist.txt --jsonl_mode shared
```

