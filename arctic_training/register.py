import inspect
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Type
from typing import Union, Tuple

if TYPE_CHECKING:
    from arctic_training.checkpoint import CheckpointEngine
    from arctic_training.config import Config
    from arctic_training.data import DataSetLoader
    from arctic_training.trainer import Trainer

_supported_dataset_registry: Dict[str, Dict[str, Type["DataSetLoader"]]] = defaultdict(
    dict
)
_supported_trainer_registry: Dict[str, Type["Trainer"]] = {}
_supported_config_registry: Dict[str, Type["Config"]] = {}
_supported_checkpoint_registry: Dict[str, Type["CheckpointEngine"]] = {}

RegistryClassTypes = Union[
    Type["DataSetLoader"], Type["Trainer"], Type["Config"], Type["CheckpointEngine"]
]


def get_dataset_type_registry(dataset_type: str) -> Dict[str, Type["DataSetLoader"]]:
    global _supported_dataset_registry
    if dataset_type not in _supported_dataset_registry:
        raise ValueError(f"Dataset type {dataset_type} is not supported.")
    return _supported_dataset_registry[dataset_type]


def get_dataset_class(dataset_type: str, dataset_name: Union[str,Tuple[str,str]]) -> Type["DataSetLoader"]:
    global _supported_dataset_registry
    if isinstance(dataset_name, tuple):
        dataset_name=dataset_name[0]        
    if dataset_name not in _supported_dataset_registry[dataset_type]:
        raise ValueError(
            f"Dataset {dataset_name} with type {dataset_type} is not supported."
        )
    return _supported_dataset_registry[dataset_type][dataset_name]


def get_config_class(
    config_name_or_class: Union[str, Type["Config"]]
) -> Type["Config"]:
    global _supported_config_registry
    if isinstance(config_name_or_class, str):
        config_name = config_name_or_class
    else:
        config_name = config_name_or_class.__name__
    if config_name not in _supported_config_registry:
        raise ValueError(f"Config {config_name} is not supported.")
    return _supported_config_registry[config_name]


def get_trainer_class(trainer_name: str) -> Type["Trainer"]:
    global _supported_trainer_registry
    if trainer_name not in _supported_trainer_registry:
        raise ValueError(f"Trainer {trainer_name} is not supported.")
    return _supported_trainer_registry[trainer_name]


def get_checkpoint_class(checkpoint_name: str) -> Type["CheckpointEngine"]:
    global _supported_checkpoint_registry
    if checkpoint_name not in _supported_checkpoint_registry:
        raise ValueError(f"Checkpoint {checkpoint_name} is not supported.")
    return _supported_checkpoint_registry[checkpoint_name]


def _validate_method_definition(
    cls: RegistryClassTypes, method_name: str, method_params: List[str] = []
) -> None:
    # Avoid circular import
    from arctic_training.checkpoint.checkpoint import CheckpointEngine
    from arctic_training.config.config import Config
    from arctic_training.data.loader import DataSetLoader
    from arctic_training.trainer.trainer import Trainer

    BaseClasses = [DataSetLoader, Trainer, Config, CheckpointEngine]

    for subclass in cls.__mro__:
        # Skip the base class itself
        if subclass in BaseClasses:
            break
        if method_name not in subclass.__dict__:
            continue
        method = subclass.__dict__[method_name]
        if not callable(method):
            raise ValueError(f"{cls.__name__}.{method_name} must be a callable method.")
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        params_names = set(p.name for p in params)
        if not params_names == set(method_params):
            raise ValueError(
                f"{cls.__name__}.{method_name} must accept exactly {method_params} as parameters, but got {params_names}."
            )
    else:
        raise ValueError(
            f"{cls.__name__} must implement its own '{method_name}' method."
        )


def _validate_class_attribute_set(cls: RegistryClassTypes, attribute: str) -> None:
    if not getattr(cls, attribute, None):
        raise ValueError(f"{cls.__name__} must define {attribute} attribute.")


def register_dataset(
    cls: Type["DataSetLoader"], force: bool = False
) -> Type["DataSetLoader"]:
    global _supported_dataset_registery

    _validate_class_attribute_set(cls, "dataset_name")
    _validate_class_attribute_set(cls, "dataset_type")
    _validate_method_definition(cls, "load_fn", ["self", "num_proc", "eval"])
    _validate_method_definition(
        cls, "tokenize_fn", ["self", "dataset", "tokenizer", "data_config"]
    )

    if cls.dataset_name in _supported_dataset_registry[cls.dataset_type] and not force:
        raise ValueError(
            f"Dataset {cls.dataset_name} with type {cls.dataset_type} is already registered. If you want to overwrite, set force=True."
        )

    _supported_dataset_registry[cls.dataset_type][cls.dataset_name] = cls
    return cls


def register_config(cls: Type["Config"], force: bool = False) -> Type["Config"]:
    global _supported_config_registry

    if cls.__name__ in _supported_config_registry and not force:
        raise ValueError(
            f"Config {cls.__name__} is already registered. If you want to overwrite, set force=True."
        )

    _supported_config_registry[cls.__name__] = cls
    return cls


def register_trainer(cls: Type["Trainer"], force: bool = False) -> Type["Trainer"]:
    # Avoid circular import
    from arctic_training.config.config import Config

    global _supported_trainer_registry

    _validate_class_attribute_set(cls, "config_type")
    _validate_class_attribute_set(cls, "dataset_type")

    if cls.__name__ in _supported_trainer_registry and not force:
        raise ValueError(
            f"Trainer {cls.__name__} is already registered. If you want to overwrite, set force=True."
        )

    if isinstance(cls.config_type, str):
        _ = get_config_class(cls.config_type)
    elif issubclass(cls.config_type, Config):
        config_cls = register_config(cls.config_type, force=force)
        cls.config_type = config_cls.__name__
    else:
        raise ValueError(
            f"Trainer {cls.__name__} must define a valid config class or a config class name."
        )

    _supported_trainer_registry[cls.__name__] = cls
    return cls


def register_checkpoint(
    cls: Type["CheckpointEngine"], force: bool = False
) -> Type["CheckpointEngine"]:
    global _supported_checkpoint_registry

    _validate_class_attribute_set(cls, "checkpoint_type")
    _validate_method_definition(cls, "save", ["self"])
    _validate_method_definition(cls, "load", ["self"])

    if cls.__name__ in _supported_checkpoint_registry and not force:
        raise ValueError(
            f"Checkpoint {cls.__name__} is already registered. If you want to overwrite, set force=True."
        )

    _supported_checkpoint_registry[cls.checkpoint_type] = cls
    return cls
