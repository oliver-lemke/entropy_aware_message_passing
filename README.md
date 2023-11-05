# GNN Deep Learning Project
Repository for 2023 Deep Learning course project of
[Oliver Lemke](https://github.com/oliver-lemke), [Phillip Nazari](), [Artiom Gesp](), [Davide Guidobene]()

## Dependencies
The main dependencies of this project are
```yaml
python: 3.9
cuda: 11.8
```

You can set up a conda environment as follows :
```bash
git clone git@github.com:oliver-lemke/gnn_dl.git
conda env create -f environment.yml
# setup user.yaml (seem configs/template.yaml)
cd source
python -m main
```

Now, set up configs/user.yaml. For instructions on how to do so, see
configs/template.yaml.
For starters, simply add your project path in the project_root_dir.
Any settings set in user.yaml will overwrite base.yaml.

You can run the project with
```bash
cd source
python -m main
```

Overall project setup can be run with
```bash
python -m scripts.setup_script
```

## Overall Setup
```
project_root_dir/                                   <--- root directory of the project
├── source/                                         <--- all code stored here
│   ├── main.py                                     <--- contains the main method
│   ├── trainer/                                    <--- contains the trainer classes responsible for training
│   │   ├── base_trainer.py                         <--- basic gnn trainer for node classification
│   │   └── ...
│   ├── datasets/
│   │   ├── __init__.py                             <--- DatasetFactory for getting dataset based on config
│   │   ├── base.py                                 <--- abstract dataset class acting as base for others
│   │   ├── planetoid.py                            <--- example dataset
│   │   └── ...
│   ├── models/
│   │   ├── __init__.py                             <--- contains the ModelFactory which is responsible for building a model
│   │   ├── basic_gcn.py                            <--- simple GCN model 
│   │   └── ...
│   ├── scripts/                                    <--- contains scripts to be run independently (e.g. for setup)
│   │   ├── setup_script.py                         <--- one script do the entire setup, does not do user.yaml config
│   │   └── ...
│   ├── utils/
│   │   ├── configs.py                              <--- ease of use class for accessing config
│   │   ├── eval_metrics.py                         <--- additional metrics to keep track of
│   │   ├── logs.py                                 <--- project-specific logging configuration
│   │   └── ...
│   └── ...
│
├── configs/
│   ├── base.yaml                                   <--- base config file used for changing the actual project
│   ├── template.yaml                               <--- template config for setting up user.yaml
│   └── user.yaml                                   <--- personal config file to set up config for this specific workspace
│
├── resurces                                        <- contains any data
│   ├── data/                                       <--- contains any used datasets
│   │   ├── README.md                               <--- markdown file which explains the data and structure
│   │   └── ...
│   │
│   ├── logs/                                       <--- contains logs
│   │   └── ...
│   │
│   ├── pretrained_weights/                         <--- contains model_weights
│   │   ├── template_weights/                       <--- template configuration
│   │   │   ├── weights.pth                         <--- actual weights for the model
│   │   │   └── pretrained_metadata.pickle          <--- metadata (config used for pretraining)
│   │
│   ├── output/                                     <--- any model output
│   │   ├── template_output/
│   │   │   ├── checkpoints/
│   │   │   │   ├── weights.pth                     <--- model weights at checkpoint
│   │   │   │   └── optimizer.pth                   <--- optimizer state at checkpoint
│   │   │   ├── best_checkpoints/
│   │   │   └── tensorboard/                        <--- tensorboard directory
│   │   │   └── wandb/                              <--- wandb directory
│   │
│   ├── cache/                                      <--- any local caching that is needed
│   │   └── ...
│
├── .github/                                        
│   ├── workflows/                                  <--- github actions 
│   │   ├── black.yml
│   │   ├── isort.yml
│   │   ├── pylint.yml
│   │   └── ...
│
├── .gitignore                                      <--- global .gitignore
├── environment.yaml                                <--- conda env file for GPU machines
├── environment_cpu.yaml                                <--- conda env file for non-GPU machines
└── README.md
```

# GitHub Actions
This project uses [black](https://pypi.org/project/black/) and
[isort](https://pypi.org/project/isort/) for formatting, and
[pylint](https://pypi.org/project/pylint/) for linting.

## PyCharm setup
1. Download the [File Watchers](https://www.jetbrains.com/help/pycharm/using-file-watchers.html)
Plugin
2. Under Settings > Tools > File Watcher > + > \<custom>: setup a new watcher for each
   1. black
      - Name: Black Watcher
      - File type: Python
      - Scope: Project Files
      - Program: \$PyInterpreterDirectory\$/black
      - Arguments: \$FilePath\$
      - Output paths to refresh: \$FilePath\$
      - Working directory: \$ProjectFileDir\$
      - Additional: as wished
   2. isort
      - Name: iSort Watcher
      - Program: \$PyInterpreterDirectory\$/isort
      - Arguments: \$FilePath\$ --sp \$ContentRoot\$/.style/.isort.cfg --settings-path \$ProjectFileDir\$/pyproject.toml
   3. pylint
      - Name: PyLint Watcher
      - Program: \$PyInterpreterDirectory\$/pylint
      - Arguments: --msg-template="\$FileDir\$/{path}:{line}:{column}:{C}:({symbol}){msg}" \$FilePath\$ --rcfile \$ProjectFileDir\$/pyproject.toml