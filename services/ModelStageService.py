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
    """
    A service for optimizing the workflow of model building stages in machine learning projects.
    This class is designed to streamline the process of handling each stage, especially when
    certain tasks, such as model training, can take a significant amount of time.

    In typical workflows, when returning to a previous point in the process, it is often necessary
    to re-run all prior stages, which can be time-consuming. This class aims to mitigate that issue.
    For each stage (notebook), an instance of this class is created, and the current and previous
    stage names are passed to it. The instance allows:

    - Running the previous stage or retrieving data from it if it has already been executed and saved.
    - Executing or retrieving results from specific heavyweight functions at each stage.
    - Saving necessary data to be passed to the subsequent stage.

    Attributes:
        previous_stage_name (str): The name of the previous stage.
        current_stage_name (str): The name of the current stage.
        data_base_path (str): The path to the base data directory.
        snapshots_data_base_path (str): The path to the directory for snapshot data (saved intermediate results).
        stages_data_base_path (str): The path to store data for each stage.
        stages_base_path (str): The path to the main directory containing stage files.
        stages (dict): A dictionary containing metadata for each stage, including notebook filenames, subfolder names, and execution order.
    """

    def __init__(self, current_stage_name=None, previous_stage_name=None):
        """
         Initializes the ModelStageService object with the given stage names
         and sets the default paths for the data and notebooks.

         Args:
             current_stage_name (str, optional): The name of the current stage.
             previous_stage_name (str, optional): The name of the previous stage.

         Raises:
             ValueError: If the current stage order is not greater than the previous stage order.
         """
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
                raise ValueError("Invalid stages order!")

    def run(self):
        """
         Runs the notebook associated with the last stage in the pipeline. Starts a chain of stages starting from
         the last one. And since at the beginning of each stage the data of the previous one is loaded,
         then if the previous stage is not executed and there is no saved data for it, it will be executed first and then
         the necessary data will be loaded.

         Raises:
             Exception: If an error occurs during the execution of the notebook.
         """
        last_stage = max(self.stages.values(), key=lambda x: x['order'])
        notebook_name = last_stage['notebook']
        notebook_path = os.path.join(self.stages_base_path, notebook_name)

        try:
            # Run notebook
            print(f"Running {notebook_name}...")
            pm.execute_notebook(
                input_path=notebook_path,
                output_path=notebook_path
            )
            print(f"{notebook_name} completed.")
        except Exception as e:
            print(f"An occurred exception during executing notebook {notebook_path}: {e}")

    def write_stage_data(self, *args):
        """
        Writes the data for the current stage into a snapshot file.

        Args:
            *args: Data to be saved. Can be a single object or multiple objects.

        Raises:
            Exception: If the current stage name is not specified.
        """
        self.check_current_stage()

        snapshot_file_path = f'{self.stages_data_base_path}{self.current_stage_name}_stage_data.pkl'
        with open(snapshot_file_path, "wb") as file:
            if len(args) == 1:
                pickle.dump(args[0], file)
            else:
                pickle.dump(args, file)

    def run_or_load_stage_data(self, reload_stage=True):
        """
        Loads data from the previous stage's snapshot or executes the corresponding notebook
        to regenerate the data. If the previous stage is not executed and there is no saved data for it, it will be
        executed first and then the necessary data will be loaded.

        Args:
            reload_stage (bool): If True, the notebook is executed to regenerate data;
                                  otherwise, loads from the snapshot if available.

        Returns:
            DataFrame: The data for the current stage.

        Raises:
            Exception: If the snapshot is missing or the data file is not saved during stage processing.
        """
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
        Runs a custom data processing function or loads data from a snapshot, depending on
        whether the snapshot exists and if it should be recreated (function will be re-executed).

        Args:
            snapshot_name (str): Name of the snapshot file.
            processing_function (callable): Function to generate the data if the snapshot does not exist.
            *args: Arguments to pass to the processing function.
            recreate_snapshot (bool): If True, always regenerates the snapshot.

        Returns:
            Data: The processed data.

        Raises:
            Exception: If the snapshot file cannot be created or loaded.
        """
        self.check_current_stage()

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

    def check_current_stage(self):
        """
        Checks if the `current_stage_name` is set. Raises an exception if it is not.

        Raises:
            Exception: If `current_stage_name` is None.
        """
        if self.current_stage_name is None:
            raise Exception('Current stage name must be specified!')
