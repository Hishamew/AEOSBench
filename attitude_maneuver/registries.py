from todd.bases.registries import Registry


class CallbackRegistry(Registry):
    pass


class LossRegistry(CallbackRegistry):
    pass


class MonitorRegistry(CallbackRegistry):
    pass


class LoggerRegistry(CallbackRegistry):
    pass


class BenchmarkRegistry(Registry):
    pass
