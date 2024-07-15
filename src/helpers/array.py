from functools import cmp_to_key


class ArrayHelper:

    @staticmethod
    def clean_list(array):
        return list(filter(None, array))

    @staticmethod
    def sort(array, compare_fn):
        return sorted(array, key=cmp_to_key(compare_fn))
