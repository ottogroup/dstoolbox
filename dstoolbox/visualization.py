"""Helper functions for creating visualizations.

Note:

* The helper functions contain additional dependencies not covered by
dstoolbox.
* They are not covered by tests and thus should only be used for
convenience but not for production purposes.

"""

import io

from sklearn.utils import murmurhash3_32

from dstoolbox.utils import get_nodes_edges


COLORS = [
    '#75DF54', '#B3F1A0', '#91E875', '#5DD637', '#3FCD12',
    '#4A88B3', '#98C1DE', '#6CA2C8', '#3173A2', '#17649B',
    '#FFBB60', '#FFDAA9', '#FFC981', '#FCAC41', '#F29416',
    '#C54AAA', '#E698D4', '#D56CBE', '#B72F99', '#B0108D',
]


def _get_shape(est):
    if hasattr(est, 'steps'):
        shape = 'invhouse'
    elif hasattr(est, 'transformer_list'):
        shape = 'oval'
    else:
        shape = 'record'
    return shape


def _get_label(name, short_name):
    if short_name:
        label = name.split('__')[-1]
    else:
        label = name.split('__')
        label[-1] = label[-1].upper()
        label = '\n'.join(label)
    return label


def _get_hex_color(est):
    hashed = int(murmurhash3_32(est)) % len(COLORS)
    return COLORS[hashed]


def _make_node(name, est, short_name=True):
    """Create a pydotplus.Node based on an sklearn estimator."""
    import pydotplus

    label = _get_label(name, short_name=short_name)
    label_type = repr(type(est)).strip("'<>").rsplit('.')[-1]
    label += '\n({})'.format(label_type)
    shape = _get_shape(est)

    return pydotplus.Node(
        name,
        label=label,
        shape=shape,
        color=_get_hex_color(label_type),
        style='filled',
    )


def make_graph(name, model, short_name=True):
    """Create a pydotplus graph of an (sklearn) Pipeline.

    Parameters
    ----------
    name : string
      Name of the model

    model : sklearn.pipeline.Pipeline
      The (sklearn) Pipeline or FeatureUnion.

    short_name : bool (default=True)
      Whether to label nodes only by the actual name of the step or by
      full name (i.e. the name returned by `get_params`).

    Returns
    -------
    graph : pydotplus.graphviz.Dot
      The pydotplus Graph

    """
    import pydotplus

    nodes, edges = get_nodes_edges(name, model)
    graph = pydotplus.Dot('Pipeline', graph_type='digraph')

    pydot_nodes = {}
    for k, v in nodes.items():
        node = _make_node(k, v, short_name=short_name)
        graph.add_node(node)
        pydot_nodes[k] = node

    for edge0, edge1 in edges:
        graph.add_edge(
            pydotplus.Edge(
                pydot_nodes[edge0],
                pydot_nodes[edge1],
            ))

    return graph


def save_graph_to_file(graph, filename):
    """Save a visualization of a pydotplus Graph to a file."""
    ext = filename.rsplit('.', 1)[-1]
    with io.open(filename, 'wb') as f:
        f.write(graph.create(format=ext))
