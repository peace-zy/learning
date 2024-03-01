import inspect
class ReformPointToCFactory(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def ClassDict(self):
        class_list = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        class_dict = {_name: _class for _name, _class in class_list}
        return class_dict

    def get_object(self, ReformPointName, **kwargs):
        if ReformPointName not in self.ClassDict:
            raise NotImplementedError('class {} has not been implement'.format(ReformPointName))
        class_obj = self.ClassDict[ReformPointName](**kwargs)
        return class_obj
