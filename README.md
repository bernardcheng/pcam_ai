# pcam_ai
## SUTD AI Project (PCAM)

### Usage:

To run train & test at the same time:

```bash
python train_test.py 
```

- ''-i', '--idx', type=int, required=False, default=1, help="Index number when saving graphs and model")

- '-e', '--epochs', type=int, required=False, default=10, help="Number of epochs for train & val")
- '-lr', '--learning_rate', type=float, required=False, default=1e-4, help="Learning rate for optimizer")

- '-b', '--batch', type=int, required=False, default=32, help="Number of batch size")

- '-d', '--data', type=str, required=False, default='/home/ubuntu/data/patchcamelyon', help="Dataset path")

- '-ts', '--train_size', type=int, required=False, default=2e18, help="Number of train dataset")
- '-vs', '--val_size', type=int, required=False, default=2e15, help="Number of train dataset")

- '-o', '--output', type=str, required=False, default='output', help="Directory for output")

- '-l', '--limit', type=bool, required=False, default=True, help="Limit GPU usage to avoid out of memory")



To test on a model:

```bash
python test.py
```

- '-b', '--batch', type=int, required=False, default=32, help="Number of batch size")

- '-d', '--data', type=str, required=False, default='/home/ubuntu/data/patchcamelyon', help="Dataset path")

- '-m', '--model', type=str, required=False, default='model_10_epochs.h5', help="Dataset path")

- '-vs', '--val_size', type=int, required=False, default=2e15-1, help="Number of train dataset")

- '-o', '--output', type=str, required=False, default='output', help="Directory for output")

- '-l', '--limit', type=bool, required=False, default=True, help="Limit GPU usage to avoid out of memory")