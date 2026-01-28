from .calc import ClippedNdarrayIterator
from .DB import DatabaseWriter
import numpy as np
import os


class Main:
    """
    Класс для обработки набора данных NDVI из каталога и записи результатов в базу данных.

    Атрибуты
    --------
    base_dir : str
        Путь к каталогу с исходными файлами для обработки.
    gpkg_path : str
        Путь к базе данных GeoPackage (GPKG), куда будут сохраняться результаты.

    Методы
    ------
    start():
        Запускает процесс обработки всех файлов из каталога base_dir.
        Для каждого файла создаётся итератор ClippedNdarrayIterator, который
        поэтапно выдаёт numpy-массивы с метаданными и идентификаторами.
        Вычисляется среднее значение NDVI с учётом маски отсутствующих данных,
        и результат сохраняется в базу данных через DatabaseWriter.
    """
    def __init__(self, base_dir, gpkg_path):
        self.base_dir = base_dir
        self.gpkg_path = gpkg_path

    def start(self):
        """
        Обрабатывает все файлы из base_dir, извлекает среднее NDVI для каждого массива
        с учётом маски отсутствующих данных, и записывает результаты в базу данных.

        Процесс:
        - Для каждого файла в base_dir создаётся объект ClippedNdarrayIterator.
        - Итератор возвращает кортеж (ndarray, meta, fid, culture).
        - Значения, равные nodata, маскируются.
        - Вычисляется среднее значение NDVI по маске.
        - Если среднее значение не ноль, оно сохраняется в базу с соответствующим fid, датой и культурой.
        - После обработки всех файлов соединение с базой закрывается.
        """
        base_dir = self.base_dir
        gpkg_path = self.gpkg_path

        writer = DatabaseWriter(gpkg_path)

        for name in os.listdir(base_dir):
            full_path = os.path.join(base_dir, name)

            iterator = ClippedNdarrayIterator(full_path, gpkg_path)

            for ndarray, meta, fid, culture in iterator:
                nodata = meta["nodata"]
                masked = np.ma.masked_equal(ndarray, nodata)
                mean_val = masked.mean()

                if hasattr(mean_val, "item"):
                    mean_val = mean_val.item()

                if mean_val == 0.0:
                    continue

                writer.insert_results_to_db(fid, iterator.get_date(), culture, mean_val)

        writer.close_db()