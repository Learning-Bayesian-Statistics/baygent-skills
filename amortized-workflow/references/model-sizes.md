# Model Sizes for Different Architectures

**ALWAYS start with the Base configuration.** Scale up to Large or XL ONLY if in-silico diagnostics show poor recovery or calibration after sufficient training. Oversized networks waste compute, train slower, and can hurt calibration on simple problems.

## Summary networks

**SetTransformer is reserved for set-based data.** Never use this architecture when dealing with time series data. If time indices (continuous or discrete) are part of the simulator output, always indicate the `time_axis` when initializing a `TimeSeriesTransformer` which is important for the internal `Time2Vec` embedding.

**Important `summary_dim` heuristic:** All summary networks accept a `summary_dim` argument that controls the dimensionality of the learned summary statistics vector. As a starting heuristic, **set `summary_dim` to 3x the number of parameters being inferred.** For example, if you are estimating 5 parameters, start with `summary_dim=15`. Scale up if diagnostics show poor recovery.

### SetTransformer
| Model Size | embed_dims                | num_heads           | num_seeds |
|------------|---------------------------|---------------------|-----------|
| Small      | (32, 32)                  | (2, 2)              | 4         |
| Base       | (64, 64, 64)              | (4, 4, 4)           | 8         |
| Large      | (128, 128, 128, 128)      | (8, 8, 8, 8)        | 16        |
| XL         | (256, 256, 256, 256, 256) | (8, 8, 8, 8, 8)     | 32        |

### FusionTransformer
| Model Size | embed_dims                | num_heads           | template_dim |
|------------|---------------------------|---------------------|--------------|
| Small      | (32, 32)                  | (2, 2)              | 64           |
| Base       | (64, 64, 64)              | (4, 4, 4)           | 128          |
| Large      | (128, 128, 128, 128)      | (8, 8, 8, 8)        | 256          |
| XL         | (256, 256, 256, 256, 256) | (8, 8, 8, 8, 8)     | 512          |

### TimeSeriesTransformer
| Model Size | embed_dims                | num_heads           | time_embed_dim |
|------------|---------------------------|---------------------|----------------|
| Small      | (32, 32)                  | (2, 2)              | 4              |
| Base       | (64, 64, 64)              | (4, 4, 4)           | 8              |
| Large      | (128, 128, 128, 128)      | (8, 8, 8, 8)        | 16             |
| XL         | (256, 256, 256, 256, 256) | (8, 8, 8, 8, 8)     | 32             |

### ConvolutionalNetwork
| Model Size | widths                  | blocks_per_stage            |
|------------|-------------------------|-----------------------------|
| Small      | (16, 32)                | 2                           |
| Base       | (32, 64, 128)           | 3                           |
| Large      | (32, 64, 128, 256)      | 4                           |
| XL         | (32, 64, 128, 256, 512) | 5                           |

### TimeSeriesNetwork
| Model Size | filters           | kernel_sizes | strides    | recurrent_dim |
|------------|-------------------|--------------|------------|---------------|
| Small      | (16, 32)          | (3, 3)       | (1, 1)     | 64            |
| Base       | (32, 64)          | (3, 3)       | (1, 1)     | 128           |
| Large      | (32, 64, 128)     | (3, 3, 3)    | (1, 1, 1)  | 256           |
| XL         | (64, 128, 256)    | (3, 3, 3)    | (1, 1, 1)  | 512           |

> **Note:** All strides are 1 — no temporal downsampling is performed before the recurrent layer, preserving full temporal resolution. This can become a bottleneck on very long sequences. If sequence length exceeds a few thousand steps, consider using strides > 1 in early conv stages to reduce the sequence length fed to the recurrent layer.

## Inference networks

### Subnet Sizing for Diffusion-Like Networks (Preferred Default)

All free- (diffusion-like) inference networks ( `FlowMatching`, `DiffusionModel`, `StableConsistencyModel`) use a `TimeMLP` subnet by default. These should be preferred to coupling flows (i.e., normalizing flows) unless the user dictates otherwise. Control their capacity via `subnet_kwargs` and `time_embedding_dim`, e.g.:

```python
bf.networks.FlowMatching(subnet_kwargs={"widths": (128,)*3, "time_embedding_dim": 16})
```

| Model Size | widths                 | time_embedding_dim |
|------------|------------------------|--------------------|
| Small      | (128, 128, 128)                  | 16       |
| Base       | (256, 256, 256, 256)             | 32       |
| Large      | (512, 512, 512, 512, 512)        | 64       |
| XL         | (512, 512, 512, 512, 512, 512)   | 64       |

### Subnet Sizing for Coupling Flows (Legacy Default)

`CouplingFlow` uses an `MLP` subnet by default. The most important hyperparameters are `subnet_kwargs`, `depth`, and `transform`, e.g.:

```python
bf.networks.CouplingFlow(subnet_kwargs={"widths": (128, 128)}, depth=2, transform="spline")
```

Use `transform="spline"` for problems with fewer than 30 parameters. For problems with more than 30 parameters, use `transform="affine"`.

| Model Size | Affine `widths` | Affine `depth` | Spline `depth` |
| ---------- | --------------- | -------------- | -------------- |
| Small      | `(64, 64)`    | 6                | 2              |
| Base       | `(128, 128)`  | 8                | 4              |
| Large      | `(256, 256)`  | 10               | 6              |
| XL         | `(512, 512)`  | 12               | 8              |
