import os

class IMDB(object):

    def __init__(self, benchmark_name, image_subset_name, dataset_path, cache_path=None):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param dataset_path: dataset path store images and image lists
        :param cache_path: store cache and proposal data
        """
        self.benchmark_name =  benchmark_name
        self.image_subset_name = image_subset_name
        self.name = benchmark_name + '_' + image_subset_name
        self.dataset_path = dataset_path
        if cache_path:
            self._cache_path = cache_path
        else:
            self._cache_path = dataset_path

        # abstract attributes
        self.num_images = 0

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self._cache_path,'{}_{}_cache'.format(self.benchmark_name, self.image_subset_name))
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path