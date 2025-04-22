# Dataset Directory

This directory is used to store hieroglyph training and test data. The actual data files are not tracked in Git due to their size, but the directory structure is maintained.

## Expected Contents

When training the model, this directory should contain labeled hieroglyph images organized by class, with the following structure:

```
Dataset/
├── train/
│   ├── A1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── A2/
│   │   └── ...
│   └── ...
└── test/
    ├── A1/
    │   └── ...
    └── ...
```

You can download the dataset from the project website or generate it using the preprocessing scripts. 