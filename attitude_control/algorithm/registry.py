__all__ = [
    'ModelRegistry',
    'MonitorRegistry',
    'EvaluatorRegistry',
]
from todd.bases.registries import Registry


class ModelRegistry(Registry):
    pass


class MonitorRegistry(Registry):
    pass


class EvaluatorRegistry(Registry):
    pass


class BenchmarkRegistry(Registry):
    pass


class EnvironmentRegistry(Registry):
    pass
