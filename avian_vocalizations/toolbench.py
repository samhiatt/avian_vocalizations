from collections import defaultdict

def get_module_versions(variables):
    """ Convenience function to print the versions of all modules in the provided dict of variables.
    Args:
        variables (dictionary of variables): A dictionary of variables, like those returned from `args()` or `globals()`.
    Returns:
        dict of sets: A dictionary with module names as keys and sets of version strings as values.
    """
    res=defaultdict(set)
    for sym in variables.keys():
        v = variables[sym]
        if hasattr(v, '__package__'):
            if v.__package__: v = __import__(v.__package__)
        if hasattr(v,'__module__'):
            v = __import__(v.__module__)
        if hasattr(v, '__version__'):
            #print("{0: >12}  {1}".format(vars[sym].__name__, vars[sym].__version__)) 
            res[v.__name__].add(v.__version__)
    return dict(res)

def print_module_versions(variables):
    """ Pretty-print the output from `get_module_versions()`
    Args:
        variables (dictionary of variables): A dictionary of variables, like those returned from `args()` or `globals()`.
    """
    versions = get_module_versions(variables)
    for key in versions:
        print("{0: >20}  {1}".format(key, versions[key])) 