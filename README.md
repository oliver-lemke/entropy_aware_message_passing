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
# setup user.yaml (see configs/template.yaml)
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

## Config
To make changes to the configuration of the project, please change configs/base.yaml.
The hyperparameters present base.yaml are the ones used in the project.
Please refer to comments within base.yaml for specific explanations.

## Overall Setup
```
project_root_dir/                                   <--- root directory of the project
├── source/                                         <--- all code stored here
│   ├── main.py                                     <--- contains the main method
│   ├── trainer/                                    <--- contains the trainer classes responsible for training
│   │   ├── base_trainer.py                         <--- basic gnn trainer for node classification
│   │   └── ...
│   ├── tester/                                     <--- contains the tester for the 2D grid testing
│   │   └── base_tester.py
│   ├── datasets/
│   │   ├── __init__.py                             <--- DatasetFactory for getting dataset based on config
│   │   ├── base.py                                 <--- abstract dataset class acting as base for others
│   │   ├── planetoid.py                            <--- Plnetoid dataset
│   │   └── ...
│   ├── models/
│   │   ├── model_utils/                            <--- utility classes for creating models
│   │   ├── __init__.py                             <--- contains the ModelFactory which is responsible for building a model
│   │   ├── basic_gcn.py                            <--- simple GCN model 
│   │   ├── entropic_gcn.py                         <--- entropic GCN model 
│   │   ├── pairnorm_gcn.py                         <--- pairnorm GCN model 
│   │   ├── g2.py                                   <--- G2 model
│   │   └── ...
│   ├── physics/                                    <--- contains scripts to be run independently (e.g. for setup)
│   │   └── physics.py/                             <--- contains energy / entropy calculations
│   ├── scripts/                                    <--- contains scripts to be run independently (e.g. for setup)
│   │   ├── hp_tuning/                              <--- scripts for hyperparameter tuning
│   │   ├── wandb/                                  <--- scripts for plotting with wandb
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
├── resources                                        <- contains any data
│   ├── data/                                       <--- contains any used datasets
│   ├── logs/                                       <--- contains logs
│   ├── pretrained_weights/                         <--- contains model_weights
│   ├── output/                                     <--- any model output
│   ├── cache/                                      <--- any local caching that is needed
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
├── environment_cpu.yaml                            <--- conda env file for non-GPU machines
└── README.md
```