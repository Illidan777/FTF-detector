import os
import pickle

import pandas as pd
import papermill as pm

EDA_STAGE = "EDA"
PREPROCESSING_STAGE = "PREPROCESSING"
FEATURE_ENGINEERING_STAGE = "FEATURE_ENGINEERING"
MODEL_BUILDING_AND_EVALUATING_STAGE = "MODEL_BUILDING_AND_EVALUATING"
MODEL_EVALUATING_STAGE = "MODEL_EVALUATING"


class ModelStageService:
    def __init__(self, current_stage_name, previous_stage_name=None):
        self.previous_stage_name = previous_stage_name
        self.current_stage_name = current_stage_name
        self.data_base_path = '../../data/'
        self.snapshots_data_base_path = '../../data/snapshots/'
        self.stages_data_base_path = '../../data/stages/'
        self.stages_base_path = '../../model/stages/'
        self.stages = {
            EDA_STAGE: {
                "notebook": "1_EDA.ipynb",
                "subFolder": "eda",
                "order": 0
            },
            PREPROCESSING_STAGE: {
                "notebook": "2_data_preprocessing.ipynb",
                "subFolder": "data_preprocessing/",
                "order": 1
            },
            FEATURE_ENGINEERING_STAGE: {
                "notebook": "3_feature_engineering.ipynb",
                "subFolder": "feature_engineering/",
                "order": 2
            },
            MODEL_BUILDING_AND_EVALUATING_STAGE: {
                "notebook": "4_model_building_and_evaluating.ipynb",
                "subFolder": "model_building/",
                "order": 3
            }
        }

        if previous_stage_name is not None:
            if self.stages[current_stage_name]["order"] <= self.stages[previous_stage_name]["order"]:
                print('Error')
                raise ValueError("Invalid stages order!")

    def run(self):
        pass

    def write_stage_data(self, *args):
        snapshot_file_path = f'{self.stages_data_base_path}{self.current_stage_name}_stage_data.pkl'
        with open(snapshot_file_path, "wb") as file:
            if len(args) == 1:
                pickle.dump(args[0], file)
            else:
                pickle.dump(args, file)

    def run_or_load_stage_data(self, reload_stage=True):
        stage_name = self.previous_stage_name

        if stage_name is None:
            return pd.read_parquet(f'{self.data_base_path}transactions.parquet', engine='pyarrow')

        snapshot_file_path = f"{self.stages_data_base_path}{stage_name}_stage_data.pkl"
        # Determine if we need to recreate the step
        print(f'Reload stage {reload_stage} path {snapshot_file_path} exists {os.path.exists(snapshot_file_path)}')
        if not reload_stage and os.path.exists(snapshot_file_path):
            # Load from checkpoint
            print(f"Loading {stage_name} from snapshot.")
            # Load from snapshot
            with open(snapshot_file_path, "rb") as file:
                print(f"Loading {stage_name} from snapshot.")
                return pickle.load(file)

        # If checkpoint is not available or recreation is required, run the processing function
        notebook_name = self.stages[stage_name]['notebook']
        notebook_path = os.path.join(self.stages_base_path, notebook_name)
        # Run notebook
        print(f"Running {notebook_name}...")
        pm.execute_notebook(
            input_path=notebook_path,
            output_path=notebook_path
        )
        print(f"{notebook_name} completed.")

        if os.path.exists(snapshot_file_path):
            print(f"Loading {stage_name} from snapshot.")
            # Load from snapshot
            with open(snapshot_file_path, "rb") as file:
                print(f"Loading {stage_name} from snapshot.")
                return pickle.load(file)
        else:
            raise Exception(
                f'At the {stage_name} stage, the data file was not saved, check that you are saving the data file at this stage')

    def run_or_load_snapshot_data(
            self,
            snapshot_name: str,
            processing_function,
            *args,
            recreate_snapshot=True
    ):
        """
        Runs a data processing step with snapshot support.

        Parameters:
        - step_name (str): Name of the step, used for checkpoint file naming.
        - processing_function (function): The function that processes the data if the checkpoint is not found.
        - *args: Arguments for the processing_function.
        - recreate_step (bool): If True, will recreate this step even if checkpoint exists.
        - global_recreate (bool): If True, will recreate all steps regardless of individual recreate_step flags.

        Returns:
        - The result of processing_function, either loaded from checkpoint or freshly computed.
        """
        sub_folder_name = self.stages[self.current_stage_name]['subFolder']
        snapshot_folder = os.path.join(self.snapshots_data_base_path, sub_folder_name)
        snapshot_file = os.path.join(snapshot_folder, f"{snapshot_name}_snapshot.pkl")

        # Create subfolder if it doesn't exist
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)
            print(f"Created snapshot folder: {snapshot_folder}")

        # Determine if we need to recreate the step
        if os.path.exists(snapshot_file) and not recreate_snapshot:
            # Load from snapshot
            with open(snapshot_file, "rb") as file:
                print(f"Loading {snapshot_name} from snapshot.")
                return pickle.load(file)

        # If snapshot is not available or recreation is required, run the processing function
        print(f"Processing {snapshot_name}...")
        result = processing_function(*args)

        # Save to checkpoint
        with open(snapshot_file, "wb") as file:
            pickle.dump(result, file)

        return result
