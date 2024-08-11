from pydantic import BaseModel, conint
from pydantic_settings import BaseSettings, SettingsConfigDict
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource
from enum import Enum
from typing import Union

class TrackingAlgorithm(str, Enum):
    DEEPOCSORT = 'DEEPOCSORT'
    OCSORT = 'OCSORT'
    DEEPSORT = 'DEEPSORT'

class DeepOcSortConfig(BaseModel):
    model_weights: str
    device: str
    fp16: bool
    per_class: bool
    det_thresh: float
    max_age: int
    min_hits: int
    iou_threshold: float
    delta_t: int
    asso_func: str
    inertia: float
    w_association_emb: float
    alpha_fixed_emb: float
    aw_param: float
    embedding_off: bool
    cmc_off: bool
    aw_off: bool
    new_kf_off: bool

class OcSortConfig(BaseModel):
    det_thresh: float
    max_age: int
    min_hits: int
    asso_threshold: float
    delta_t: int
    asso_func: str
    inertia: float
    use_byte: bool

class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: conint(ge=1, le=65536) = 6379
    stream_id: str
    input_stream_prefix: str = 'featureextractor'
    output_stream_prefix: str = 'objecttracker'

class DeepSortConfig(BaseModel):
    max_cosine_distance: float
    min_confidence: float
    max_iou_distance: float
    max_age: int
    n_init: int

class ObjectTrackerConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    tracker_config: Union[DeepOcSortConfig, OcSortConfig,DeepSortConfig]
    tracker_algorithm: TrackingAlgorithm
    redis: RedisConfig
    prometheus_port: conint(gt=1024, le=65536) = 8000

    model_config = SettingsConfigDict(env_nested_delimiter='__')

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)