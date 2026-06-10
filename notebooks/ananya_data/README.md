# Ananya Gupta notebook data

Local copies of all files read by `Ananya_Gupta_final.ipynb`.

```
chat/
  BBW8512a.cha                  # sample CHAT transcript (Cinderella task)
manifests/
  train/manifest.jsonl          # speaker-disjoint train split
  test/manifest.jsonl           # speaker-disjoint test split
dora/
  pre_post_train_eval_metrics.json
team_transcription_results.json   # final-report Table tab:main (held-out test metrics)
logit_bias_smoke/
  logit_bias_summary.json
  disfluency_prob_traces_bias0.json
  dev_predictions_bias=0.jsonl
qa_results/
  metrics_base_single.json              # audio only
  metrics_base_gold_transcripts.json    # audio + gold transcript
  metrics_base_with_finetuned_transcripts.json
```

Note: manifest rows still reference audio paths under `/scratch/merged_data_processed/`; only metadata is bundled here.
