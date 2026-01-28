from pyproj import Transformer
from shapely.ops import transform
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import os
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import re
from .DB import DatabaseReader
from numpy import ndarray


class NDVICalculator:
    """
   Класс для вычисления NDVI (Normalized Difference Vegetation Index) из спутниковых снимков.

   Parameters
   ----------
   path_dir : str
       Путь к директории с файлами снимков.

   Attributes
   ----------
   ndarray : numpy.ndarray
       Рассчитанный массив NDVI с размерностью (1, height, width).
   profile : dict
       Метаданные растрового изображения, соответствующие NDVI.
   """
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.ndarray, self.profile = self.calculate_ndvi()

    @staticmethod
    def define_bands_dict(path):
        """
        Определяет пути к спектральным диапазонам в директории.

        Parameters
        ----------
        path : str
            Путь к директории со снимками.

        Returns
        -------
        dict
            Словарь с ключами 'blue', 'green', 'red', 'nir', 'swir' и путями к соответствующим файлам (str или None).

        Raises
        ------
        Exception
            Если возникает ошибка при определении файлов каналов.
        """
        try:
            bands = {'blue': None, 'green': None, 'red': None, 'nir': None, 'swir': None}
            for file in os.listdir(path):
                full_path = os.path.join(path, file)
                if file.endswith('B2.tif'):
                    bands['blue'] = full_path
                elif file.endswith('B3.tif'):
                    bands['green'] = full_path
                elif file.endswith(('B4.tif', 'B04_10m.tif')):
                    bands['red'] = full_path
                elif file.endswith(('B5.tif', 'B08_10m.tif')):
                    bands['nir'] = full_path
                elif file.endswith(('B6.tif', 'B7.tif')):
                    bands['swir'] = full_path
            return bands

        except Exception as e:
            print("Ошибка определения снимков:", e)
            raise

    def calculate_ndvi(self):
        """
        Рассчитывает NDVI на основе спектральных данных.

        Returns
        -------
        tuple
            - ndarray : numpy.ndarray
                Массив NDVI с дополнительной размерностью (1, height, width).
            - profile : dict
                Метаданные растрового изображения с обновлённым типом, драйвером и nodata.

        Raises
        ------
        Exception
            При ошибках чтения файлов или вычисления.
        """
        try:
            bands = self.define_bands_dict(self.path_dir)

            with rasterio.open(bands['red']) as red_src:
                red = red_src.read(1).astype('float32')
                profile = red_src.profile

            with rasterio.open(bands['nir']) as nir_src:
                nir = nir_src.read(1).astype('float32')

            ndvi = (nir - red) / (nir + red + 1e-10)
            profile.update(dtype=rasterio.float32, driver="GTiff", nodata=0, count=1)

            ndarray = ndvi[np.newaxis, ...]

            return ndarray, profile


        except Exception as e:
            print("Ошибка вычисления NDVI:", e)
            raise

    def get_date(self):
        """
        Извлекает дату из имени файла по шаблону '20YYYYMMDD'.

        Returns
        -------
        str or None
            Дата в формате 'YYYY-MM-DD' или None, если дата не найдена.

        Raises
        ------
        Exception
            При ошибках чтения файлов или парсинга.
        """
        try:
            filename = os.listdir(self.path_dir)[0]
            match = re.search(r"(20\d{6})", filename)
            if not match:
                return None
            date_str = match.group(1)
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        except Exception as e:
            print("Ошибка определения дат:", e)
            raise

class CloudMasker(NDVICalculator):
    """
    Класс для наложения маски облаков на рассчитанный NDVI.

    Наследует NDVICalculator.

    Parameters
    ----------
    path_dir : str
        Путь к директории со снимками.

    Attributes
    ----------
    source : str or None
        Источник маски облаков ('sentinel', 'landsat' или None).
    path_cloud_img : str or None
        Путь к файлу маски облаков.
    ndarray : numpy.ndarray
        Массив NDVI с применённой маской облаков.
    profile : dict
        Метаданные с применённой маской.
    """
    def __init__(self, path_dir):
        super().__init__(path_dir)
        self.source = None
        self.path_cloud_img = self.define_cloud_img(path_dir)

        self.ndarray, self.profile = self.apply_mask(self.ndarray, self.profile)

    @staticmethod
    def extract_bit(arr, bit):
        """
        Извлекает указанный бит из целочисленного массива.

        Parameters
        ----------
        arr : numpy.ndarray
            Массив целых чисел.
        bit : int
            Индекс бита для извлечения.

        Returns
        -------
        numpy.ndarray
            Массив битовых значений (0 или 1).
        """
        return (arr >> bit) & 1

    def define_cloud_img(self, path_dir):
        """
        Определяет путь к файлу маски облаков в директории.

        Parameters
        ----------
        path_dir : str
            Путь к директории со снимками.

        Returns
        -------
        str or None
            Путь к файлу маски облаков или None, если не найден.

        Raises
        ------
        Exception
            При ошибках обхода файлов.
        """
        try:
            for root, _, files in os.walk(path_dir):
                for file in files:
                    if "SCL" in file and "aux" not in file:
                        self.source = "sentinel"
                        return os.path.join(root, file)
                    if "QA_PIXEL" in file and "aux" not in file:
                        self.source = "landsat"
                        return os.path.join(root, file)
            return None

        except Exception as e:
            print("Ошибка определения маски облаков:", e)
            raise

    @staticmethod
    def _align_mask(mask: ndarray, src_profile, dst_profile):
        """
        Подгоняет маску облаков под целевой профиль растрового изображения.

        Parameters
        ----------
        mask : numpy.ndarray
            Исходная маска облаков.
        src_profile : dict
            Профиль исходного растрового изображения маски.
        dst_profile : dict
            Профиль целевого растрового изображения.

        Returns
        -------
        numpy.ndarray
            Маска, приведённая к размеру и пространственной привязке целевого изображения.

        Raises
        ------
        Exception
            При ошибках ресемплинга.
        """
        try:
            if (src_profile["width"], src_profile["height"]) == (dst_profile["width"], dst_profile["height"]):
                return mask

            sample = np.empty((dst_profile["height"], dst_profile["width"]), dtype=mask.dtype)

            reproject(
                source=mask,
                destination=sample,
                src_transform=src_profile["transform"],
                src_crs=src_profile["crs"],
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                resampling=Resampling.nearest
            )
            return sample

        except Exception as e:
            print("Ошибка трансформации маски облаков:", e)
            raise

    def mask_clouds_sentinel(self, ndarray, profile):
        """
        Применяет маску облаков для данных Sentinel.

        Parameters
        ----------
        ndarray : numpy.ndarray
            Массив NDVI.
        profile : dict
            Профиль растрового изображения.

        Returns
        -------
        tuple
            - ndarray : numpy.ndarray
                Массив NDVI с применённой маской облаков.
            - profile : dict
                Обновлённый профиль.

        Raises
        ------
        Exception
            При ошибках чтения и обработки маски.
        """
        try:
            mask_values = [3, 7, 8, 9]

            img = ndarray

            with rasterio.open(self.path_cloud_img) as src:
                clouds = src.read(1)
                clouds = self._align_mask(clouds, src.profile, profile)

                cloud_mask = np.isin(clouds, mask_values)
                img_masked = np.where(~cloud_mask, img, 0)

                return img_masked, profile

        except Exception as e:
            print("Ошибка маскирования Sentinel:", e)
            raise

    def mask_clouds_landsat(self, ndarray, profile):
        """
        Применяет маску облаков для данных Landsat.

        Parameters
        ----------
        ndarray : numpy.ndarray
            Массив NDVI.
        profile : dict
            Профиль растрового изображения.

        Returns
        -------
        tuple
            - ndarray : numpy.ndarray
                Массив NDVI с применённой маской облаков.
            - profile : dict
                Обновлённый профиль.

        Raises
        ------
        Exception
            При ошибках чтения и обработки маски.
        """
        try:
            DILATED_CLOUD = 1
            CLOUD_BIT = 3
            CLOUD_SHADOW_BIT = 4
            WATER = 7

            img = ndarray

            with rasterio.open(self.path_cloud_img) as src:
                clouds = src.read(1)
                clouds = self._align_mask(clouds, src.profile, profile)

            cloud_mask = (
                    self.extract_bit(clouds, CLOUD_BIT)
                    | self.extract_bit(clouds, CLOUD_SHADOW_BIT)
                    | self.extract_bit(clouds, DILATED_CLOUD)
                    | self.extract_bit(clouds, WATER)
            )

            cloud_mask = (cloud_mask == 0)
            img_masked = np.where(cloud_mask, img, 0)

            return img_masked, profile

        except Exception as e:
            print("Ошибка маскирования Landsat:", e)
            raise

    def apply_mask(self, ndarray, profile):
        """
        Применяет маску облаков, выбирая метод в зависимости от источника данных.

        Parameters
        ----------
        ndarray : numpy.ndarray
            Массив NDVI.
        profile : dict
            Профиль растрового изображения.

        Returns
        -------
        tuple
            - ndarray : numpy.ndarray
                Массив NDVI с применённой маской облаков.
            - profile : dict
                Обновлённый профиль.
        """
        if self.source == "sentinel":
            return self.mask_clouds_sentinel(ndarray, profile)
        return self.mask_clouds_landsat(ndarray, profile)

class ClippedNdarrayIterator(CloudMasker, DatabaseReader):
    """
    Итератор по обрезанным NDVI-массивам по геометриям из базы данных.

    Наследует CloudMasker и DatabaseReader.

    Parameters
    ----------
    path_dir : str
        Путь к директории со спутниковыми снимками.
    path_db : str
        Путь к файлу базы данных SQLite/GeoPackage.
    table_name : str, optional
        Имя таблицы с геометриями (по умолчанию "fields").

    Attributes
    ----------
    polygon_crs : pyproj.CRS
        Система координат полигонов из базы данных.
    raster_crs : pyproj.CRS
        Система координат растровых данных.
    ndvi_dataset_masked : rasterio.io.DatasetReader
        Объект растрового датасета с маскированным NDVI.
    _gen : list
        Список сгенерированных кортежей (ndarray, meta, fid, culture).
    """
    def __init__(self, path_dir, path_db, table_name="fields"):
        CloudMasker.__init__(self, path_dir)
        DatabaseReader.__init__(self, path_db, table_name)

        self.polygon_crs = self.get_polygon_crs()
        self.raster_crs = self.profile["crs"]

        self.ndvi_dataset_masked = self.rasterio_dataset(self.ndarray, self.profile)

        self._gen = [(ndarray, meta, fid, culture) for ndarray, meta, fid, culture in self._img_generator()]

    @staticmethod
    def rasterio_dataset(ndarray, profile):
        """
        Создаёт временный растровый датасет из массива и профиля в памяти.

        Parameters
        ----------
        ndarray : numpy.ndarray
            Массив данных.
        profile : dict
            Профиль растрового изображения.

        Returns
        -------
        rasterio.io.DatasetReader
            Временный растровый датасет в памяти.

        Raises
        ------
        Exception
            При ошибках создания датасета.
        """
        try:
            mem = MemoryFile()
            dataset = mem.open(**profile)
            dataset.write(ndarray)
            return dataset

        except Exception as e:
            print("Ошибка Rasterio dataset:", e)
            raise

    def _img_generator(self):
        """
        Генерирует обрезанные NDVI-массивы для каждого полигона из базы.

        Yields
        ------
        tuple
            - ndarray : numpy.ndarray
                Обрезанный NDVI-массив.
            - meta : dict
                Метаданные для обрезанного растрового изображения.
            - fid : int or str
                Идентификатор полигона.
            - culture : str
                Название культуры.

        Raises
        ------
        Exception
            При ошибках трансформации, маскирования и чтения из базы.
        """
        try:
            transformer = Transformer.from_crs(
                self.polygon_crs,
                self.raster_crs,
                always_xy=True)

            for geom, fid, culture in self.geom_generator():
                geom_projected = transform(transformer.transform, geom)
                geojson = geom_projected.__geo_interface__

                ndarray, out_transform = mask(self.ndvi_dataset_masked, [geojson], crop=True)
                meta = self.ndvi_dataset_masked.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": ndarray.shape[1],
                    "width": ndarray.shape[2],
                    "transform": out_transform,
                    "nodata": 0
                })
                yield ndarray, meta, fid, culture

            self.close_db()

        except Exception as e:
            print("Ошибка генерации обрезанных изображений:", e)
            raise

    def __iter__(self):
        """
        Возвращает итератор по сгенерированным обрезанным изображениям.

        Returns
        -------
        iterator
            Итератор, возвращающий кортежи (ndarray, meta, fid, culture).
        """
        return iter(self._gen)

    def __del__(self):
        self.close_db()