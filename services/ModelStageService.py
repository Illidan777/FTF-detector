import os
import pickle

import pandas as pd
import papermill as pm

EDA_STAGE = "EDA"
PREPROCESSING_STAGE = "PREPROCESSING"
FEATURE_ENGINEERING_STAGE = "FEATURE_ENGINEERING"
MODEL_BUILDING_STAGE = "MODEL_BUILDING"
MODEL_EVALUATING_STAGE = "MODEL_EVALUATING"


class ModelStageService:
    def __init__(self, current_stage_name, previous_stage_name=None):
        self.previous_stage_name = previous_stage_name
        self.current_stage_name = current_stage_name
        self.data_base_path = '../data/'
        self.snapshots_data_base_path = '../data/snapshots/'
        self.stages_data_base_path = '../data/stages/'
        self.stages_base_path = '../model/stages/'
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
            MODEL_BUILDING_STAGE: {
                "notebook": "4_model_building.ipynb",
                "subFolder": "model_building/",
                "order": 3
            },
            MODEL_EVALUATING_STAGE: {
                "notebook": "5_model_evaluating.ipynb",
                "subFolder": "model_evaluating/",
                "order": 4
            }
        }
        if (previous_stage_name is not None) & (
                self.stages[current_stage_name]["order"] <= self.stages[previous_stage_name]["order"]):
            raise ValueError("Invalid stages order!")

    def run(self):
        pass
        
    def write_stage_data(self, df):
        df.to_parquet(f"{self.stages_data_base_path}{self.current_stage_name}_stage_data.parquet", engine='pyarrow')

    def run_or_load_stage_data(self, reload_stage=True):
        stage_name = self.previous_stage_name

        if stage_name is None:
            return pd.read_parquet(f'{self.data_base_path}transactions.parquet', engine='pyarrow')

        stage_data_file = f"{self.stages_data_base_path}{stage_name}_stage_data.parquet"
        # Determine if we need to recreate the step
        if not reload_stage and os.path.exists(stage_data_file):
            # Load from checkpoint
            with open(stage_data_file, "rb") as file:
                print(f"Loading {stage_name} from checkpoint.")
                return pd.read_parquet(stage_data_file, engine='pyarrow')

        # If checkpoint is not available or recreation is required, run the processing function
        notebook_name = self.stages[stage_name]['notebook']
        notebook_path = os.path.join(self.stages_base_path, notebook_name)
        # Run notebook
        print(f"Running {notebook_name}...")
        pm.execute_notebook(
            notebook_path,
            notebook_path.replace(".ipynb", "_output.ipynb")
        )
        print(f"{notebook_name} completed.")

        if os.path.exists(stage_data_file):
            with open(stage_data_file, "rb") as file:
                print(f"Loading {stage_name} from checkpoint.")
                return pd.read_parquet(stage_data_file, engine='pyarrow')
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
        Runs a data processing step with checkpoint support.

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
        checkpoint_file = f"{self.snapshots_data_base_path}{sub_folder_name}{snapshot_name}_snapshot.pkl"

        # Determine if we need to recreate the step
        if os.path.exists(checkpoint_file) and not recreate_snapshot:
            # Load from checkpoint
            with open(checkpoint_file, "rb") as file:
                print(f"Loading {snapshot_name} from checkpoint.")
                return pickle.load(file)

        # If checkpoint is not available or recreation is required, run the processing function
        print(f"Processing {snapshot_name}...")
        result = processing_function(*args)

        # Save to checkpoint
        with open(checkpoint_file, "wb") as file:
            pickle.dump(result, file)

        return result
