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
```

## Overall Setup
```
project_root_dir/                                   <--- root directory of the project
├── source/                                         <--- all code stored here
│   ├── main.py                                     <--- contains the main method
│   ├── trainer.py                                  <--- contains the trainer class responsible for all trainin 
│   │   ├── datasets/
│   │   │   ├── dataset_template.py                 <--- template for how to write a dataset
│   │   │   └── ...
│   ├── models/
│   │   ├── __init__.py                             <--- contains the model_factory which is responsible for building a model
│   │   ├── template_model.py                       <--- template for how a model should look like
│   │   ├── specialized_networks/                   <--- use this folder for special changes to the network
│   │   │   ├── special_example.py                  <--- example for such a network change
│   │   │   └── ...
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
├── data/                                           <--- contains any used datasets
│   ├── README.md                                   <--- markdown file which explains the data and structure
│   └── ...
│
├── logs/                                           <--- contains logs
│   └── ...
│
├── pretrained_weights/                             <--- contains model_weights
│   ├── template_weights/                           <--- template configuration
│   │   ├── weights.pth                             <--- actual weights for the model
│   │   └── pretrained_metadata.pickle              <--- metadata (config used for pretraining)
│
├── output/                                         <--- any model output
│   ├── template_output/
│   │   ├── checkpoints/
│   │   │   ├── weights.pth                         <--- model weights at checkpoint
│   │   │   └── optimizer.pth                       <--- optimizer state at checkpoint
│   │   ├── best_checkpoints/
│   │   └── tensorboard/                            <--- tensorboard directory
│   │   └── wandb/                                  <--- wandb directory
│
├── cache/                                          <--- any local caching that is needed
│   └── ...
│
├── .github/                                        
│   ├── workflows/                                  <--- github actions 
│   │   ├── black.yml
│   │   ├── isort.yml
│   │   ├── pylint.yml
│   │   └── ...
│
├── .gitignore                                      <--- global .gitignore
├── requirements.txt
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