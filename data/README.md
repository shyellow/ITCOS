# Study inputs

The original terrain-tolerance meshes cover the soil erosion study area in Ganxian District, Jiangxi, China. Each mesh consists of single-ring PolygonZ triangles.

| TIN tolerance | Raw mesh | Triangles |
|---|---|---:|
| 5 m | `tin/rongcha5m.shp` | 65,448 |
| 10 m (default) | `tin/rongcha10.shp` | 26,861 |
| 15 m | `tin/rongcha15m.shp` | 15,572 |
| 30 m | `tin/rongcha30.shp` | 5,815 |

Each shapefile includes its SHX, DBF, PRJ and CPG companions. The default 10 m mesh has 14,180 shared XY vertices and elevations from 135 to 792 m. For the exact commands to select a mesh, see the [reproduction guide](../README.md).

**Shared RKCP.** `RKCP.tif` contains R × K × C × P for all four meshes. It is a single-band Float32 GeoTIFF with 426 columns × 458 rows, 30 m × 30 m native pixels and 120,676 valid cells. NoData is tagged in the band.

All layers use WGS 84 / UTM zone 50N (EPSG:32650). Mesh XY and Z are in metres. TIN tolerance describes terrain approximation. The workflow samples band 1 of RKCP on its 30 m grid by the inverse affine transform, retaining valid zeros. With dimensionless LS, `RKCP × LS` inherits the supplied product's numerical scale. The study class table uses tonnes per square kilometre per year. Supply RKCP on that scale for a new experiment.

`../SHA256SUMS` records the hashes of all four meshes and the single shared raster. The historical checkpoints in `../reference/checkpoints.json` and the precomputed products in `../reference/` apply to the 10 m example.

## Using a different dataset

- Supply a continuous mesh of single-part, closed PolygonZ triangles. Each shared XY vertex must have consistent Z across its incident faces. The raw layer must use a projected CRS with metre units and metric elevations. Keep the original feature order when reproducing an existing experiment.
- Supply `.shp`, `.shx`, `.dbf`, `.prj` and UTF-8 `.cpg` together. The input requires a unique integer `Tri_Index`, a numeric `Slope_Deg` field, and geometry fields `Shape_Area` and `Shape_Leng`. The terrain stage recomputes slope from XYZ and adds `ORIG_FID`, `SLen_m`, `LS` and other terrain fields. Start with the raw schema to avoid output field collisions.
- Supply a single-band RKCP raster in the same CRS as the TIN, with a valid mask or tagged NoData value. At least one original face needs four valid samples for the optimization objective.
