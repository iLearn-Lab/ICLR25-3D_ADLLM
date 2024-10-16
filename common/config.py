import json
import logging
from common.registry import registry
from typing import Dict


from omegaconf import OmegaConf


class LLMConfig:
    def __init__(self, args):
        self.config = {}
        self.args = args

        config = OmegaConf.load(self.args.cfg_path)

        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config)
        dataset_config = self.build_dataset_config(config)

        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config
        )

    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration file."

        model_cls = registry.get_model_class(model.arch)
        assert model_cls is not None, f"Model '{model.arch}' has not been registered."

        model_type = kwargs.get("model.model_type", None)
        if not model_type:
            model_type = model.get("model_type", None)
        # else use the model type selected by user.

        assert model_type is not None, "Missing model_type."

        model_config_path = model_cls.default_config_path(model_type=model_type)

        model_config = OmegaConf.create()
        # hiararchy override, customized config > default config
        model_config = OmegaConf.merge(
            model_config,
            OmegaConf.load(model_config_path),
            {"model": config["model"]},
        )

        return model_config

    @staticmethod
    def build_runner_config(config):
        return {"run": config.run}

    @staticmethod
    def build_dataset_config(config):
        datasets = config.get("train_datasets", None)
        if datasets is None:
            raise KeyError(
                "Expecting 'datasets' as the root key for dataset configuration."
            )
        eval_datasets = config.get("eval_datasets", None)
        if eval_datasets is None and config.run.evaluate:
            raise KeyError(
                "Expecting 'eval_datasets' when evaluate is True."
            )

        return {"train_datasets": config.train_datasets, "eval_datasets": config.eval_datasets}


    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def train_datasets_cfg(self):
        return self.config.train_datasets
    
    @property
    def eval_datasets_cfg(self):
        return self.config.eval_datasets

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        # logging.info("\n======  Dataset Attributes  ======")
        # datasets = self.config.datasets

        # for dataset in datasets:
        #     if dataset in self.config.datasets:
        #         logging.info(f"\n======== {dataset} =======")
        #         dataset_config = self.config.datasets[dataset]
        #         logging.info(self._convert_node_to_json(dataset_config))
        #     else:
        #         logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        # logging.info(f"\n======  Model Attributes  ======")
        # logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)