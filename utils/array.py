from functools import cmp_to_key


class ArrayHelper:

    @staticmethod
    def clean_list(array):
        return list(filter(None, array))

    @staticmethod
    def sort(array, compare_fn):
        return sorted(array, key=cmp_to_key(compare_fn))

    @staticmethod
    def object_list_to_dict(object_list, property_key):
        res_dict = {}
        for item in object_list:
            res_dict[getattr(item, property_key)] = item
        return res_dict
