"""Use this to allow imports, e.g.

```
from dstoolbox.transformers import ItemSelector
```

instead of

```
from dstoolbox.transformers.slicing import ItemSelector
```

"""
from dstoolbox.data.load import load_w2v_format
from dstoolbox.data.load import load_w2v_vocab


__all__ = ['load_w2v_format', 'load_w2v_vocab']
