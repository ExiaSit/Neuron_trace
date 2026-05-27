# Neuron_trace

## Pipeline modes

Default behavior is unchanged:

```bash
python run_pipeline.py --mode neuron
```

To let batches flow across stages, set the per-stage concurrent task cap:

```bash
python run_pipeline.py --mode stage-batch --stage-max-tasks 20
```

In `stage-batch` mode, stage 1 runs up to 20 neurons, passes that completed batch to downstream stages, then starts the next 20 while downstream stages work on previous batches. `merge` waits until the soma crop/seg branch and neurite seg branch are both complete for the same batch.
