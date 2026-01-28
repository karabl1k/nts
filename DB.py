import sqlite3
import os
import re
from shapely import wkb


class DatabaseReader:
    """
    Класс для чтения геометрий и атрибутов из GeoPackage / SQLite базы данных.

    Используется для извлечения полигонов, их идентификаторов и атрибутов
    (например, культуры), а также для определения системы координат.

    Parameters
    ----------
    path_db : str
        Путь к файлу базы данных (GeoPackage или SQLite).
    table_name : str
        Имя таблицы с геометриями.

    Attributes
    ----------
    path_db : str
        Путь к базе данных.
    table_name : str
        Имя таблицы с геометриями.
    conn : sqlite3.Connection
        Соединение с базой данных.
    cursor : sqlite3.Cursor
        Курсор для выполнения SQL-запросов.
    """
    def __init__(self, path_db, table_name):
        self.path_db = path_db
        self.table_name = table_name
        self.conn = sqlite3.connect(path_db)
        self.cursor = self.conn.cursor()

    def get_polygon_crs(self):
        """
        Получает систему координат полигонов из метаданных GeoPackage.

        Returns
        -------
        str
            Система координат в формате 'ORG:EPSG_CODE'
            (например, 'EPSG:4326').

        Raises
        ------
        Exception
            При ошибках выполнения SQL-запросов.
        """
        try:
            self.cursor.execute(f"SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = '{self.table_name}'")
            srs_id = self.cursor.fetchone()[0]
            self.cursor.execute(f"SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id = {srs_id}")
            org, epsg_code = self.cursor.fetchone()
            return f"{org}:{epsg_code}"

        except Exception as e:
            print("Ошибка получения системы координат:", e)
            raise

    def geom_generator(self):
        """
        Генератор геометрий и атрибутов из таблицы базы данных.

        Yields
        ------
        tuple
            - geom : shapely.geometry.base.BaseGeometry
                Геометрия полигона.
            - fid : int
                Идентификатор объекта.
            - culture : str
                Название культуры или атрибут полигона.

        Raises
        ------
        Exception
            При ошибках чтения данных или декодирования геометрии.
        """
        try:
            self.cursor.execute(f"SELECT geom, fid, culture FROM {self.table_name}")
            for row in self.cursor:
                geom = wkb.loads(row[0][40:])
                fid = row[1]
                culture = row[2]
                yield geom, fid, culture

        except Exception as e:
            print("Ошибка вычисления геометрий:", e)
            raise

    def close_db(self):
        if self.conn:
            self.conn.close()

class DatabaseWriter:
    """
    Класс для записи результатов NDVI в базу данных GeoPackage / SQLite.

    Создаёт таблицу результатов и записывает агрегированные значения NDVI
    для каждого полигона и даты.

    Parameters
    ----------
    path_db : str
        Путь к файлу базы данных.
    layer_name : str, optional
        Имя слоя с полигонами. Если не задано, определяется автоматически.
    fid_column : str, optional
        Имя столбца с идентификаторами полигонов. Если не задано, определяется автоматически.
    table_name : str, optional
        Имя таблицы для записи результатов (по умолчанию "results").

    Attributes
    ----------
    path_db : str
        Путь к базе данных.
    table_name : str
        Имя таблицы с результатами NDVI.
    cursor : sqlite3.Cursor
        Курсор для выполнения SQL-запросов.
    conn : sqlite3.Connection
        Соединение с базой данных.
    layer_name : str
        Имя слоя с полигонами.
    fid_column : str
        Имя столбца с идентификаторами полигонов.
    """
    def __init__(self, path_db: str, layer_name: str = None, fid_column: str = None, table_name = "results"):
        self.path_db = path_db
        self.table_name = table_name
        self.cursor, self.conn = self.connect_to_db(path_db)
        self.layer_name = layer_name or self.detect_polygons_layer()
        self.fid_column = fid_column or self.detect_fid_column(self.layer_name)
        self.create_ndvi_table()

    @staticmethod
    def connect_to_db(path_db):
        """
        Подключается к базе данных и загружает расширение SpatiaLite.

        Parameters
        ----------
        path_db : str
            Путь к файлу базы данных.

        Returns
        -------
        tuple
            - cursor : sqlite3.Cursor
                Курсор базы данных.
            - conn : sqlite3.Connection
                Соединение с базой данных.

        Raises
        ------
        Exception
            При ошибках подключения или загрузки расширения.
        """
        try:
            mod_spatialite_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                r"mod_spatialite-5.1.0-win-amd64"
            )

            os.environ["PATH"] += os.pathsep + mod_spatialite_path

            conn = sqlite3.connect(path_db)
            conn.enable_load_extension(True)
            conn.load_extension(os.path.join(mod_spatialite_path, "mod_spatialite.dll"))

            cursor = conn.cursor()
            return cursor, conn

        except Exception as e:
            print("Ошибка подключения к базе данных:", e)
            raise

    @staticmethod
    def safe_name(name: str):
        """
        Проверяет корректность SQL-идентификатора.

        Parameters
        ----------
        name : str
            Имя таблицы или столбца.

        Returns
        -------
        str
            Корректное имя.

        Raises
        ------
        ValueError
            Если имя не соответствует требованиям SQL.
        """
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return name

    def detect_polygons_layer(self):
        """
        Автоматически определяет слой с полигонами в GeoPackage.

        Returns
        -------
        str
            Имя таблицы с геометриями.

        Raises
        ------
        Exception
            При ошибках SQL-запроса.
        """
        try:
            self.cursor.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features';")
            return self.cursor.fetchone()[0]

        except Exception as e:
            print("Ошибка обнаружения полигонов:", e)
            raise

    def detect_fid_column(self, table_name: str):
        """
        Определяет столбец идентификатора (FID) в таблице.

        Parameters
        ----------
        table_name : str
            Имя таблицы.

        Returns
        -------
        str
            Имя столбца с первичным ключом или первого столбца таблицы.

        Raises
        ------
        Exception
            При ошибках чтения структуры таблицы.
        """
        try:
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            cols = self.cursor.fetchall()
            for cid, name, ctype, notnull, dflt, pk in cols:
                if pk == 1:
                    return name
            return cols[0][1]

        except Exception as e:
            print("Ошибка обнаружения FID's:", e)
            raise

    def create_ndvi_table(self):
        """
        Создаёт таблицу для хранения результатов NDVI, если она не существует.

        Raises
        ------
        Exception
            При ошибках создания таблицы.
        """
        try:
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fid INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    culture TEXT NOT NULL,
                    ndvi REAL NOT NULL,
                    UNIQUE(fid, date, culture)
                );
            """)
            self.conn.commit()

        except Exception as e:
            print("Ошибка создания таблицы:", e)
            raise

    def insert_results_to_db(self, fid: int, date: str, culture: str, ndvi: float):
        """
        Записывает результат NDVI в базу данных.

        Parameters
        ----------
        fid : int
            Идентификатор полигона.
        date : str
            Дата съёмки в формате 'YYYY-MM-DD'.
        culture : str
            Название культуры.
        ndvi : float
            Значение NDVI.

        Raises
        ------
        Exception
            При ошибках записи данных.
        """
        try:
            self.cursor.execute(f"""
                INSERT OR IGNORE INTO {self.table_name} (fid, date, culture, ndvi)
                VALUES (?, ?, ?, ?)
            """, (fid, date, culture, ndvi))
            self.conn.commit()

        except Exception as e:
            print("Ошибка записи данных:", e)
            raise

    def close_db(self):
        self.conn.close()