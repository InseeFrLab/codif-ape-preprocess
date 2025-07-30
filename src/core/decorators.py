from core.registry import register_rule


def rule(name=None, tags=None, description=""):
    def decorator(func):
        register_rule(
            func=func,
            name=name or func.__name__,
            tags=tags or [],
            description=description,
        )
        return func

    return decorator
