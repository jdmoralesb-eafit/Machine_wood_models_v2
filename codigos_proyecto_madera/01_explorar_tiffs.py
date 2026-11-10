import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np
import os

tiff_dir = os.path.join(os.path.dirname(__file__), '..', 'Biomas_zonas')
output_dir = os.path.dirname(__file__)

tiff_files = sorted([f for f in os.listdir(tiff_dir) if f.endswith('.tif')])

# Bounding box de Latinoamérica y Centroamérica en EPSG:4326 (lon/lat)
LATAM_BOUNDS_4326 = (-120.0, -56.0, -30.0, 33.0)

MAX_WIDTH = 2000

for tif_name in tiff_files:
    tif_path = os.path.join(tiff_dir, tif_name)
    print(f"\n{'='*60}")
    print(f"Archivo: {tif_name}")
    print(f"{'='*60}")

    with rasterio.open(tif_path) as src:
        print(f"  CRS: {src.crs}")
        print(f"  Dimensiones totales: {src.width} x {src.height}")
        print(f"  Bandas: {src.count}")
        print(f"  Tipo de dato: {src.dtypes}")
        print(f"  NoData: {src.nodata}")

        # 1. Reproyectar el bbox de Latam al CRS del raster
        try:
            bounds_in_src_crs = transform_bounds(
                "EPSG:4326", src.crs, *LATAM_BOUNDS_4326
            )
        except Exception as e:
            print(f"  No se pudo reproyectar el bbox para este CRS: {e}. Se omite.")
            continue

        # 2. Calcular la ventana e intersectarla con el extent real del raster
        window = from_bounds(*bounds_in_src_crs, transform=src.transform)
        window = window.intersection(Window(0, 0, src.width, src.height))

        win_width = int(round(window.width))
        win_height = int(round(window.height))

        if win_width <= 0 or win_height <= 0:
            print("  Sin intersección con el bbox de Latam. Se omite.")
            continue

        # 3. Factor de reducción para no exceder MAX_WIDTH
        scale = min(1.0, MAX_WIDTH / win_width)
        out_width = max(1, int(win_width * scale))
        out_height = max(1, int(win_height * scale))

        print(f"  Ventana Latam: {win_width} x {win_height} px -> leyendo a {out_width} x {out_height}")

        data = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest
        )

        nodata = src.nodata

        if nodata is not None:
            valid = data[data != nodata]
        else:
            valid = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()

        if valid.size == 0:
            print("  Sin datos válidos en esta ventana, se omite.")
            continue

        print(f"\n  Estadísticas (Latam, sin nodata):")
        print(f"    Min: {valid.min():.4f}")
        print(f"    Max: {valid.max():.4f}")
        print(f"    Media: {valid.mean():.4f}")
        print(f"    Std: {valid.std():.4f}")
        print(f"    Valores únicos (primeros 20): {np.unique(valid)[:20]}")
        print(f"    Total píxeles válidos: {len(valid)}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{tif_name} (recorte Latam)', fontsize=14, fontweight='bold')

        masked_data = np.where(data == nodata, np.nan, data).astype(float) if nodata is not None else data.astype(float)

        im = axes[0].imshow(masked_data, cmap='viridis')
        axes[0].set_title('Mapa Latam')
        plt.colorbar(im, ax=axes[0], label='Valor')

        axes[1].hist(valid, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].set_title('Distribución de valores')
        axes[1].set_xlabel('Valor')
        axes[1].set_ylabel('Frecuencia')

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'visualizacion_{os.path.splitext(tif_name)[0]}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Gráfico guardado en: {fig_path}")