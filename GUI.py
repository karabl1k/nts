import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import rasterio
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .calc import NDVICalculator, CloudMasker, ClippedNdarrayIterator
from .analysis import NDVIPlotter
from .main import Main


class NDVICalcTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._build()

    def _build(self):
        ttk.Label(self, text="Папка со снимком").pack(anchor="w")
        ttk.Button(self, text="Выбрать папку", command=self.select_dir).pack(fill="x")
        self.lbl = ttk.Label(self, text="—")
        self.lbl.pack(anchor="w", pady=5)
        ttk.Button(self, text="Вычислить и сохранить NDVI", command=self.run).pack(pady=10)

    def select_dir(self):
        self.path = filedialog.askdirectory()
        self.lbl.config(text=self.path)

    def run(self):
        try:
            calc = NDVICalculator(self.path)
            out = filedialog.asksaveasfilename(defaultextension=".tif")
            with rasterio.open(out, "w", **calc.profile) as dst:
                dst.write(calc.ndarray)
            messagebox.showinfo("OK", "NDVI сохранён")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

class CloudMaskTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._build()

    def _build(self):
        ttk.Label(self, text="Папка со снимком").pack(anchor="w")
        ttk.Button(self, text="Выбрать папку", command=self.select_dir).pack(fill="x")
        self.lbl = ttk.Label(self, text="—")
        self.lbl.pack(anchor="w", pady=5)
        ttk.Button(self, text="Сохранить замаскированный NDVI", command=self.run).pack(pady=10)

    def select_dir(self):
        self.path = filedialog.askdirectory()
        self.lbl.config(text=self.path)

    def run(self):
        try:
            masker = CloudMasker(self.path)
            out = filedialog.asksaveasfilename(defaultextension=".tif")
            with rasterio.open(out, "w", **masker.profile) as dst:
                dst.write(masker.ndarray)
            messagebox.showinfo("OK", "Masked NDVI сохранён")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

class ClipTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._build()

    def _build(self):
        ttk.Label(self, text="Папка со снимком").pack(anchor="w")
        ttk.Button(self, text="Выбрать папку", command=self.select_img).pack(fill="x")
        self.img_lbl = ttk.Label(self, text="—")
        self.img_lbl.pack(anchor="w")

        ttk.Label(self, text="GeoPackage / SQLite").pack(anchor="w", pady=(10, 0))
        ttk.Button(self, text="Выбрать БД", command=self.select_db).pack(fill="x")
        self.gpkg_lbl = ttk.Label(self, text="—")
        self.gpkg_lbl.pack(anchor="w")

        self.save_mode = tk.StringVar(value="both")
        ttk.Label(self, text="Режим сохранения").pack(anchor="w", pady=(10, 0))

        ttk.Radiobutton(
            self,
            text="Только GeoTIFF",
            variable=self.save_mode,
            value="tif"
        ).pack(anchor="w")

        ttk.Radiobutton(
            self,
            text="Только БД",
            variable=self.save_mode,
            value="db"
        ).pack(anchor="w")

        ttk.Button(self, text="Запустить", command=self.run).pack(pady=10)

    def select_img(self):
        self.img_dir = filedialog.askdirectory()
        self.img_lbl.config(text=self.img_dir)

    def select_db(self):
        self.gpkg_path = filedialog.askopenfilename(filetypes=[("SQLite", "*.*")])
        self.gpkg_lbl.config(text=self.gpkg_path)

    def run(self):
        try:
            if self.save_mode.get() == "db":
                Main(self.img_dir, self.gpkg_path).start()
            else:
                iterator = ClippedNdarrayIterator(self.img_dir, self.gpkg_path)
                out_dir = filedialog.askdirectory() if self.save_mode.get() == "tif" else None

                date = iterator.get_date()

                for ndarray, meta, fid, culture in iterator:
                    path = os.path.join(out_dir, f"{culture}_{fid}_{date}.tif")
                    with rasterio.open(path, "w", **meta) as dst:
                        dst.write(ndarray)


            messagebox.showinfo("OK", "Clipping завершён")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

class AnalysisTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__()
        self.plotter = None
        self.processed = None
        self._build_layout()

    def _build_layout(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")

        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="SQLite база").pack(anchor="w")
        ttk.Button(left, text="Открыть .gpkg", command=self.open_db).pack(fill="x")
        self.gpkg_label = ttk.Label(left, text="не выбрана", foreground="gray")
        self.gpkg_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Culture").pack(anchor="w")
        self.culture_cb = ttk.Combobox(left, state="readonly")
        self.culture_cb.pack(fill="x")
        self.culture_cb.bind("<<ComboboxSelected>>", self.update_fids)

        ttk.Label(left, text="FID (1,2,3 или пусто = все)").pack(anchor="w")
        self.fid_entry = ttk.Entry(left)
        self.fid_entry.pack(fill="x")

        ttk.Label(left, text="Дата начала (YYYY-MM-DD)").pack(anchor="w")
        self.start_entry = ttk.Entry(left)
        self.start_entry.pack(fill="x")

        ttk.Label(left, text="Дата конца (YYYY-MM-DD)").pack(anchor="w")
        self.end_entry = ttk.Entry(left)
        self.end_entry.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=8)

        self.filter_neighbor = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Фильтр выбросов", variable=self.filter_neighbor).pack(anchor="w")

        ttk.Label(left, text="Порог соседа").pack(anchor="w")
        self.neigh_thresh = ttk.Entry(left)
        self.neigh_thresh.insert(0, "0.01")
        self.neigh_thresh.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=8)
        self.smooth = tk.BooleanVar()
        ttk.Checkbutton(left, text="Интерполяция", variable=self.smooth).pack(anchor="w")

        ttk.Label(left, text="Количество точек интерполяции").pack(anchor="w")
        self.smooth_param = ttk.Entry(left)
        self.smooth_param.insert(0, "500")
        self.smooth_param.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=8)
        self.show_raw = tk.BooleanVar(value=True)
        self.show_removed = tk.BooleanVar()
        self.show_mean = tk.BooleanVar(value=True)

        ttk.Checkbutton(left, text="Значения снимков", variable=self.show_raw).pack(anchor="w")
        ttk.Checkbutton(left, text="Удалённые", variable=self.show_removed).pack(anchor="w")
        ttk.Checkbutton(left, text="Средняя", variable=self.show_mean).pack(anchor="w")

        ttk.Button(left, text="Построить график", command=self.run).pack(fill="x", pady=10)

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Button(left, text="Экспорт графика", command=self.export_plot).pack(fill="x", pady=10)

        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True)

    def open_db(self):
        path = filedialog.askopenfilename(filetypes=[("SQLite", "*.*")])
        if not path:
            return
        self.plotter = NDVIPlotter(path)
        self.gpkg_label.config(text=path, foreground="black")
        self.culture_cb["values"] = self.plotter.get_cultures()

    def update_fids(self, *_):
        pass

    def parse_fids(self):
        txt = self.fid_entry.get().strip()
        if not txt:
            return None
        return [int(x) for x in txt.split(',') if x.strip()]

    def run(self):
        if not self.plotter:
            messagebox.showerror("Ошибка", "База не выбрана")
            return

        culture = self.culture_cb.get()
        if not culture:
            messagebox.showerror("Ошибка", "Culture не выбрана")
            return

        fids = self.parse_fids()

        try:
            processed = self.plotter.compute_data(
                culture=culture,
                fids=fids,
                filter_neighbor_flag=self.filter_neighbor.get(),
                neighbor_thresh=float(self.neigh_thresh.get()),
                smooth=self.smooth.get(),
                smooth_num=int(self.smooth_param.get()),
                start_date=self.start_entry.get() or None,
                end_date=self.end_entry.get() or None
            )

            fig, ax = self.plotter.plot_processed(
                processed,
                culture=culture,
                fids=fids,
                show_raw_points=self.show_raw.get(),
                show_removed=self.show_removed.get(),
                mean_curve=self.show_mean.get()
            )

            self.current_fig = fig

            for w in self.plot_frame.winfo_children():
                w.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def export_plot(self):
        if self.current_fig is None:
            messagebox.showerror("Ошибка", "Сначала постройте график")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return

        try:
            self.current_fig.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

class NDVIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NDVI Application")
        self.geometry("1200x800")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        nb.add(NDVICalcTab(nb), text="NDVI")
        nb.add(CloudMaskTab(nb), text="Cloud Mask")
        nb.add(ClipTab(nb), text="Clip & DB")
        nb.add(AnalysisTab(nb), text="Analysis")

if __name__ == "__main__":
    NDVIApp().mainloop()