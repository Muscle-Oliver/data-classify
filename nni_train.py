import sys
import os
import json
from clearml import Task
from nni.experiment import Experiment
task = Task.init(project_name='HPO Auto-training',
                 task_name='Datatype Classifier',
                 output_uri=True,
                 auto_connect_arg_parser=False,
                 auto_connect_frameworks=False)


class ExperimentManager():
    def __init__(self):
        search_space = {
            "lr": {"_type": "loguniform", "_value": [1e-4, 1e-2]},
            "weight_decay": {"_type": "loguniform", "_value": [5e-6, 5e-4]},
            "max_epoch": {"_type": "choice", "_value": [5, 10]},
            "batch_size": {"_type": "choice", "_value": [64, 128, 192]},
        }
        task.connect(search_space, name="HPO Search Space")
        BASE_DIR = os.path.split(os.path.realpath(__file__))[0]
        print(f"BASE_DIR: {BASE_DIR}")
        experiment = Experiment("local")
        experiment.config.experiment_name = "DataTypeClassifier"
        experiment.config.trial_command = "python train.py"
        experiment.config.trial_code_directory = BASE_DIR
        experiment.config.experiment_working_directory = f"{BASE_DIR}/nni/nni-experiments"
        experiment.config.search_space = search_space
        experiment.config.tuner.name = "TPE"
        experiment.config.tuner.class_args["optimize_mode"] = "maximize"
        experiment.config.max_trial_number = 5
        experiment.config.trial_concurrency = 1
        experiment.config.trial_gpu_number = 1
        experiment.config.tuner_gpu_indices = os.getenv('GPU_INDICES', 1)
        experiment.config.training_service.use_active_gpu = True
        
        experiment.config.debug = False
        self.experiment = experiment
        self.BASE_DIR = BASE_DIR

    def run(self, port=8788):
        print(f"Start NNI experiment at port: {port}")
        result_code = self.experiment.run(port=port)


class NNITrainer():
    def __init__(self):
        self.em = ExperimentManager()
    def run(self):
        exp_id = self.em.experiment.id
        run_flag = self.em.run()
        return run_flag


class NNIRuntimeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


if __name__ == "__main__":
    filepath = os.path.split(os.path.realpath(__file__))[0]
    preprocess = os.popen(f"cd {filepath} && python preprocess1.py")
    NNITrainer().run()
