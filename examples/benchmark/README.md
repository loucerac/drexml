# Running the benchmark

It refers to our HPC config, so it needs some adaptation to other configs.

For the CPU system:
```
bash run.sh 0
```

For the GPU system:
```
bash run.sh 1
```


Once it has been run in the `CPU` and `GPU` systems:
```
$CONDA_RUN jupyter execute 05_analyze_results.ipynb
```
