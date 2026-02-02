import csv
from collections import defaultdict
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from typing import cast
from scipy.signal import savgol_filter


class CSVExporterAll:
    """
    Класс для экспорта всех записей NDVI из базы данных в CSV-файл
    без дополнительной агрегации или фильтрации.

    Таблица вида: fid,date,culture,ndvi

    Parameters
    ----------
    db_path : str
        Путь к SQLite / GeoPackage базе данных.
    table_name : str, optional
        Имя таблицы с результатами NDVI (по умолчанию "results").
    """
    def __init__(self, db_path, table_name="results"):
        self.db_path = db_path
        self.table_name = table_name

    def export(self, csv_path):
        """
        Экспортирует все данные NDVI в CSV-файл.

        CSV содержит столбцы: fid, date, culture, ndvi.

        Parameters
        ----------
        csv_path : str
            Путь для сохранения CSV-файла.

        Raises
        ------
        Exception
            При ошибках подключения к базе данных или записи файла.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT fid, date, culture, ndvi
                FROM {self.table_name}
                ORDER BY date ASC;
            """)
            rows = cursor.fetchall()

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow(["fid", "date", "culture", "ndvi"])

                writer.writerows(rows)

            conn.close()
            print(f"CSV успешно создан: {csv_path}")

        except Exception as e:
            print("Ошибка при экспорте CSV:", e)
            raise

class CSVExporter:
    """
    Класс для экспорта NDVI в CSV с группировкой по датам и культурам.

    Поддерживает фильтрацию по диапазону дат и формирует
    табличный формат, удобный для последующего анализа.

    Таблица вида: barley_fid,barley_date,barley_ndvi

    Parameters
    ----------
    db_path : str
        Путь к базе данных.
    table_name : str, optional
        Имя таблицы с результатами NDVI (по умолчанию "results").
    """
    def __init__(self, db_path, table_name="results"):
        self.conn = sqlite3.connect(db_path)
        self.table_name = table_name

    def get_date_range(self):
        """
        Получает минимальную и максимальную даты в таблице.

        Returns
        -------
        tuple
            (min_date, max_date) в формате строк.

        Raises
        ------
        Exception
            При ошибке выполнения SQL-запроса.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT MIN(date), MAX(date) FROM {self.table_name};")
            min_date, max_date = cursor.fetchone()
            return min_date, max_date

        except Exception as e:
            print("Ошибка получения дат:", e)
            raise

    def fetch_all(self, start_date=None, end_date=None):
        """
        Извлекает данные NDVI из базы данных с фильтрацией по датам.

        Parameters
        ----------
        start_date : str, optional
            Начальная дата (YYYY-MM-DD).
        end_date : str, optional
            Конечная дата (YYYY-MM-DD).

        Returns
        -------
        list of tuple
            Список записей (fid, date, culture, ndvi).

        Raises
        ------
        Exception
            При ошибке чтения данных.
        """
        try:
            cursor = self.conn.cursor()

            query = f"SELECT fid, date, culture, ndvi FROM {self.table_name}"
            params = []

            if start_date and end_date:
                query += " WHERE date BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            elif start_date:
                query += " WHERE date >= ?"
                params.append(start_date)
            elif end_date:
                query += " WHERE date <= ?"
                params.append(end_date)

            query += " ORDER BY date ASC;"

            cursor.execute(query, params)
            return cursor.fetchall()

        except Exception as e:
            print("Ошибка получения данных из базы:", e)
            raise

    def export(self, csv_path, start_date=None, end_date=None):
        """
        Экспортирует данные NDVI в агрегированный CSV-файл.

        Parameters
        ----------
        csv_path : str
            Путь для сохранения CSV.
        start_date : str, optional
            Начальная дата.
        end_date : str, optional
            Конечная дата.

        Raises
        ------
        Exception
            При ошибках экспорта.
        """
        try:
            if start_date is None or end_date is None:
                min_date, max_date = self.get_date_range()
                if start_date is None:
                    start_date = min_date
                if end_date is None:
                    end_date = max_date

            rows = self.fetch_all(start_date=start_date, end_date=end_date)

            data_by_date_culture = defaultdict(lambda: defaultdict(list))
            cultures = set()

            for fid, date, culture, ndvi in rows:
                cultures.add(culture)
                data_by_date_culture[date][culture].append((fid, ndvi))

            cultures = sorted(list(cultures))

            header = []
            for culture in cultures:
                header.append(f"{culture}_fid")
                header.append(f"{culture}_date")
                header.append(f"{culture}_ndvi")

            table_rows = []

            sorted_dates = sorted(data_by_date_culture.keys())

            for date in sorted_dates:
                max_rows = max(len(data_by_date_culture[date].get(culture, [])) for culture in cultures)

                for i in range(max_rows):
                    row = []
                    for culture in cultures:
                        entries = data_by_date_culture[date].get(culture, [])
                        if i < len(entries):
                            fid_val, ndvi_val = entries[i]
                            row.append(fid_val)
                            row.append(date)
                            row.append(ndvi_val)
                        else:
                            row.extend(["", "", ""])
                    table_rows.append(row)

            with open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(table_rows)

            print(f"CSV успешно создан: {csv_path}")

        except Exception as e:
            print("Ошибка при экспорте CSV:", e)
            raise

class NDVIPlotterMethods:

    def __init__(self, gpkg_path):
        try:
            self.conn = sqlite3.connect(gpkg_path)
        except Exception as e:
            print("Ошибка подключения к БД:", e)
            raise

    @staticmethod
    def auto_convert(value):
        """
        Преобразует numpy-типы в стандартные Python-типы.

        Parameters
        ----------
        value : str
            Значение для преобразования.

        Returns
        -------
        any
            Преобразованное значение.
        """
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        return value

    def get_cultures(self):
        """
        Возвращает список уникальных культур в базе данных.

        Returns
        -------
        list of str
            Отсортированный список культур.
        """
        try:
            query = "SELECT DISTINCT culture FROM results"
            df = pd.read_sql_query(query, self.conn)
            cultures: list[str] = [
                str(self.auto_convert(c))
                for c in df["culture"].dropna().unique()]
            return sorted(cultures)

        except Exception as e:
            print("Ошибка получения культур:", e)
            raise

    def get_fids(self, culture=None):
        """
        Возвращает список уникальных идентификаторов полей (FID).

        Может возвращать все FID из таблицы результатов либо только
        относящиеся к указанной культуре.

        Parameters
        ----------
        culture : str, optional
            Название культуры. Если не задано, возвращаются FID для всех культур.

        Returns
        -------
        list of str
            Отсортированный список FID в строковом представлении.

        Raises
        ------
        Exception
            При ошибке выполнения SQL-запроса или обработки данных.
        """
        try:
            if culture is None:
                query = "SELECT DISTINCT fid FROM results"
                df = pd.read_sql_query(query, self.conn)
            else:
                query = "SELECT DISTINCT fid FROM results WHERE culture = :culture"
                df = pd.read_sql_query(query, self.conn, params={"culture": culture})
            fids: list[str] = [
                str(self.auto_convert(i))
                for i in df["fid"].dropna().unique()]
            return sorted(fids)

        except Exception as e:
            print("Ошибка получения FID's:", e)
            raise

    def filter_neighbor(self, df, thresh):
        """
        Фильтрует выбросы NDVI на основе соседних значений.

        Значение считается выбросом, если его отличие от среднего
        между предыдущим и следующим значением превышает заданный порог.

        Parameters
        ----------
        df : pandas.DataFrame
            Таблица с колонками ``date`` и ``ndvi``, отсортированная по дате.
        thresh : float
            Порог отклонения NDVI от среднего соседних значений.

        Returns
        -------
        tuple of pandas.DataFrame
            - Отфильтрованный DataFrame без выбросов
            - DataFrame с удалёнными значениями (``date``, ``ndvi_removed``)

        Raises
        ------
        Exception
            При ошибке обработки данных.
        """
        try:
            if len(df) < 3:
                return df.copy(), pd.DataFrame(columns=["date", "ndvi_removed"])
            removed_rows = []
            ndvi = df["ndvi"].values
            dates = df["date"].values
            mask = np.zeros(len(df), dtype=bool)
            for i in range(1, len(df) - 1):
                avg = (ndvi[i - 1] + ndvi[i + 1]) / 2
                if abs(ndvi[i] - avg) > thresh:
                    mask[i] = True
                    removed_rows.append((dates[i], ndvi[i]))
            df_filtered = df[~mask].copy().reset_index(drop=True)
            removed_df = pd.DataFrame(removed_rows, columns=["date", "ndvi_removed"])
            return df_filtered, removed_df

        except Exception as e:
            print("Ошибка фильтрации данных:", e)
            raise

    def interpolate(self, df, num=None, interpolate_window_divisor=10, polyorder=2):
        """
        Выполняет сглаживание временного ряда NDVI с помощью фильтра Савицкого-Голея.

        Parameters
        ----------
        df : pandas.DataFrame
            Таблица с колонками ``date`` и ``ndvi``.
        num : int, optional
            Количество точек в итоговом сглаженном ряду.
        window_length : int, optional
            Длина окна фильтра (должна быть нечетной).
        polyorder : int, optional
            Степень полинома для аппроксимации.
        """
        try:
            df = df[["date", "ndvi"]].dropna().copy()

            if len(df) < 5:
                raise ValueError("Слишком мало точек для выбранной степени полинома.")

            df = df.sort_values("date").drop_duplicates(subset="date")

            if len(df) < 5:
                raise ValueError("Слишком мало данных для анализа")

            full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq='D')
            temp_df = pd.DataFrame({"date": full_range})
            df = pd.merge(temp_df, df, on="date", how="left")

            y_interp = df["ndvi"].interpolate(method='linear').bfill().ffill().to_numpy()

            fraction = interpolate_window_divisor / 100
            window_length = int(len(y_interp) * fraction)
            window_length = max(3, window_length)

            if window_length % 2 == 0:
                window_length += 1
            y_interpolated = savgol_filter(y_interp, window_length=window_length, polyorder=polyorder)

            x_numeric = np.linspace(0, len(y_interpolated) - 1, num)
            y_final = np.interp(x_numeric, np.arange(len(y_interpolated)), y_interpolated)

            dates_final = pd.date_range(start=df["date"].min(), end=df["date"].max(), periods=num)

            return pd.DataFrame({
                "date": dates_final,
                "ndvi": y_final
            })

        except Exception as e:
            print(f"Ошибка сглаживания Савицкого-Голея: {e}")
            raise

class NDVIPlotter(NDVIPlotterMethods):
    """
    Класс для анализа, обработки и визуализации временных рядов NDVI.

    Класс предназначен для работы с результатами расчёта NDVI,
    сохранёнными в GeoPackage / SQLite базе данных. Он предоставляет
    инструменты для:

    - получения списка культур и идентификаторов полей (FID)
    - фильтрации выбросов NDVI
    - сглаживания временных рядов
    - объединения нескольких кривых NDVI
    - вычисления среднего временного ряда
    - визуализации и экспорта графиков

    Данные извлекаются из таблицы ``results`` и обрабатываются в формате
    ``pandas.DataFrame``.

    Parameters
    ----------
    gpkg_path : str
        Путь к базе данных.
    Notes
    -----
    Ожидается, что таблица ``results`` содержит следующие поля:

    - ``fid`` : int or str
        Идентификатор поля.
    - ``date`` : str or datetime
        Дата наблюдения.
    - ``culture`` : str
        Название культуры.
    - ``ndvi`` : float
        Значение NDVI.

    Все операции чтения выполняются в режиме только чтения.
    Класс не изменяет содержимое базы данных.

    Attributes
    ----------
    conn : sqlite3.Connection
        Активное соединение с базой данных.
    """
    def __init__(self, gpkg_path):
        super().__init__(gpkg_path)
        self.DEFAULT_PARAMS = {
            'filter_neighbor_flag': True,
            'neighbor_thresh': None,

            'interpolate': False,
            'interpolate_num': None,
            'interpolate_window_divisor': None,

            'start_date': None,
            'end_date': None,

            'show_raw_points': True,
            'show_removed': False,
            'mean_curve': True,

            'line_color': None,
            'mean_color': "red",

            'fig_size': (12, 6)
    }

    def compute_data(self, culture, fids=None, **kwargs):
        """
        Вычисляет и подготавливает временные ряды NDVI для выбранной культуры.

        Поддерживает фильтрацию выбросов, сглаживание, агрегацию
        и расчёт среднего временного ряда.

        Returns
        -------
        dict
            Словарь со следующими ключами:

            - ``fids`` : list
                Список использованных FID.
            - ``curves`` : dict
                Сглаженные или отфильтрованные кривые NDVI по FID.
            - ``raw`` : dict
                Исходные временные ряды NDVI.
            - ``removed`` : dict
                Удалённые значения NDVI.
            - ``merged`` : pandas.DataFrame or None
                Объединённая таблица всех временных рядов.
            - ``mean_series`` : pandas.Series or None
                Средний временной ряд NDVI.

        Raises
        ------
        Exception
            При ошибке чтения или обработки данных.
        """
        params = {**self.DEFAULT_PARAMS, **kwargs}

        try:
            if fids is None:
                query = f"SELECT DISTINCT fid FROM results WHERE culture='{culture}'"
                df_fids = pd.read_sql_query(query, self.conn)
                fids_list: list[str] = sorted(
                    str(self.auto_convert(i))
                    for i in df_fids["fid"].dropna().unique())
            else:
                if isinstance(fids, (list, tuple, np.ndarray)):
                    fids_list = list(fids)
                else:
                    fids_list = [fids]
            if len(fids_list) == 0:
                return {"fids": [], "curves": {}, "raw": {}, "removed": {}, "merged": None, "mean_series": None}
            curves = {}
            raw = {}
            removed = {}
            merged = None
            for fid in fids_list:
                query = f"""
                    SELECT date, ndvi
                    FROM results
                    WHERE culture='{culture}' AND fid={fid}
                    ORDER BY date
                """
                df = pd.read_sql_query(query, self.conn)
                if df.empty:
                    continue
                df["date"] = pd.to_datetime(df["date"])
                df["ndvi"] = df["ndvi"].astype(float)
                if params["start_date"]:
                    df = df[df["date"] >= pd.to_datetime(params["start_date"])]
                if params["end_date"]:
                    df = df[df["date"] <= pd.to_datetime(params["end_date"])]
                df = df.reset_index(drop=True)
                raw[fid] = df.copy()
                removed_fid = pd.DataFrame(columns=["date", "ndvi_removed"])
                if params["filter_neighbor_flag"]:
                    df, removed_tmp = self.filter_neighbor(df, params["neighbor_thresh"])
                    if removed_fid.empty:
                        removed_fid = removed_tmp.copy()
                    elif not removed_tmp.dropna(how="all").empty:
                        removed_fid = pd.concat([removed_fid, removed_tmp], ignore_index=True)
                if params["interpolate"]:
                    df = self.interpolate(df, num=params["interpolate_num"], interpolate_window_divisor=params["interpolate_window_divisor"])
                df = df.reset_index(drop=True)
                series = df.set_index("date")["ndvi"]
                series.name = f"fid_{fid}"
                if merged is None:
                    merged = series.to_frame()
                else:
                    merged = merged.join(series, how="outer")
                curves[fid] = df.copy()
                removed[fid] = removed_fid.copy()
            mean_series = None
            if merged is not None and not merged.empty:
                mean_series = merged.mean(axis=1)
            return {"fids": list(curves.keys()), "curves": curves, "raw": raw, "removed": removed, "merged": merged,
                    "mean_series": mean_series}

        except Exception as e:
            print("Ошибка вычисления данных:", e)
            raise

    def plot_processed(self, processed, culture, fids=None, **kwargs):
        """
        Строит график NDVI на основе предварительно обработанных данных.

        Returns
        -------
        tuple
            (matplotlib.figure.Figure, matplotlib.axes.Axes)

        Raises
        ------
        Exception
            При ошибке построения графика.
        """
        params = {**self.DEFAULT_PARAMS, **kwargs}

        try:
            curves = processed.get("curves", {})
            raw = processed.get("raw", {})
            removed = processed.get("removed", {})
            mean_series = cast(Series, processed.get("mean_series"))
            if fids is None:
                plot_fids = list(curves.keys())
            else:
                plot_fids = [fid for fid in fids if fid in curves]
            fig, ax = plt.subplots(figsize=params["fig_size"])
            if len(plot_fids) == 0:
                return fig, ax
            removed_plotted_label = False
            for idx, fid in enumerate(plot_fids):
                df = curves[fid]
                if params["line_color"] is None:
                    ax.plot(df["date"], df["ndvi"], linewidth=1.2)
                else:
                    ax.plot(df["date"], df["ndvi"], linewidth=1.2, color=params["line_color"])
                if params["show_raw_points"]:
                    rdf = raw.get(fid)
                    if rdf is not None and not rdf.empty:
                        ax.scatter(rdf["date"], rdf["ndvi"], s=12)
            if params["mean_curve"] and mean_series is not None:
                ax.plot(mean_series.index, mean_series.values, linewidth=3.0, color=params["mean_color"])
            if params["show_removed"]:
                for fid in plot_fids:
                    rdf = removed.get(fid)
                    if rdf is not None and not rdf.empty:
                        if not removed_plotted_label:
                            ax.scatter(rdf["date"], rdf["ndvi_removed"], s=30, color="red", marker="x", label="Removed")
                            removed_plotted_label = True
                        else:
                            ax.scatter(rdf["date"], rdf["ndvi_removed"], s=30, color="red", marker="x")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Culture: {culture}\nFIDs: {len(plot_fids)}")
            ax.set_xlabel("Date")
            ax.set_ylabel("NDVI")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=9)
            plt.tight_layout()
            return fig, ax

        except Exception as e:
            print("Ошибка построения графика:", e)
            raise

    def plot_culture(self, culture, fids:None, **kwargs):
        """
        Полный цикл обработки и построения графика NDVI для культуры.

        Returns
        -------
        tuple
            (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        params = {**self.DEFAULT_PARAMS, **kwargs}

        pracessed = self.compute_data(culture=culture, fids=fids, **params)
        return self.plot_processed(pracessed, culture, fids=fids, **params)

    def export_plot(self, culture, file_path, dpi=300, **kwargs):
        """
       Экспортирует график NDVI в файл.

       Returns
       -------
       None
       """
        params = {**self.DEFAULT_PARAMS, **kwargs}

        fig, ax = self.plot_culture(culture=culture, **params)
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)