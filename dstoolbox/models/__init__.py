"""Use this to allow imports, e.g.

```
from dstoolbox.transformers import ItemSelector
```

instead of

```
from dstoolbox.transformers.slicing import ItemSelector
```

"""
from dstoolbox.models.text import W2VClassifier


__all__ = ['W2VClassifier']
