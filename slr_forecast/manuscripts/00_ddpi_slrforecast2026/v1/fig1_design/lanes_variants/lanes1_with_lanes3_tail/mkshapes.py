"""Derive simplified Antarctica / Greenland outlines (40x40 box) from Natural Earth 110m land."""
import json, numpy as np, shapefile  # pyshp
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
import warnings; warnings.filterwarnings("ignore")
r = shapefile.Reader("/Users/minchew/.local/share/cartopy/shapefiles/natural_earth/physical/ne_110m_land.shp")
geoms = [shape(s.__geo_interface__) for s in r.shapes()]
polys = []
for g in geoms:
    polys += list(g.geoms) if isinstance(g, MultiPolygon) else [g]
ant = unary_union([p for p in polys if p.centroid.y < -60])
grl = max([p for p in polys if -75 < p.centroid.x < -10 and p.centroid.y > 60], key=lambda p: p.area)
def proj(geom, crs):
    def f(lon, lat, z=None):
        lon = np.radians(np.asarray(lon)); lat = np.radians(np.asarray(lat))
        if crs == "EPSG:3031":
            rho = 2 * np.tan(np.pi / 4 + lat / 2)
            return rho * np.sin(lon), rho * np.cos(lon)
        rho = 2 * np.tan(np.pi / 4 - lat / 2); l = lon + np.radians(45)
        return rho * np.sin(l), -rho * np.cos(l)
    return transform(f, geom)
antp = proj(ant, "EPSG:3031")   # Antarctic polar stereographic (Greenwich up? check)
grlp = proj(grl, "EPSG:3413")
def fit(geom, pad=3, tol=None, others=()):
    minx, miny, maxx, maxy = geom.bounds
    s = (40 - 2 * pad) / max(maxx - minx, maxy - miny)
    ox = pad + ((40 - 2 * pad) - s * (maxx - minx)) / 2
    oy = pad + ((40 - 2 * pad) - s * (maxy - miny)) / 2
    f = lambda x, y, z=None: (ox + s * (x - minx), oy + s * (maxy - y))
    return [transform(f, g) for g in (geom,) + tuple(others)]
def d(geom, tol=0.8):
    geom = geom.simplify(tol)
    ps = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = ""
    for p in ps:
        if p.area < 2: continue
        c = np.array(p.exterior.coords)
        out += "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in c[:-1]) + " Z "
    return out.strip()
# sub-regions by longitude wedges (in lon/lat), then projected
def wedge(l0, l1, latmax=-60):
    lons = np.linspace(l0, l1, 50)
    return Polygon([(l0, -90)] + [(l, latmax) for l in lons] + [(l1, -90)])
ap_ll = ant.intersection(Polygon([(-80, -74), (-80, -60), (-50, -60), (-50, -74)]))
wais_ll = ant.intersection(wedge(-165, -40)).difference(ap_ll)
eais_ll = ant.difference(wedge(-165, -40))
antp, ap, wa, ea = fit(antp, others=(proj(ap_ll, "EPSG:3031"), proj(wais_ll, "EPSG:3031"), proj(eais_ll, "EPSG:3031")))
grlp, = fit(grlp)
res = {"ANT": d(antp), "ap": d(ap, 0.3), "wais": d(wa), "eais": d(ea), "GRL": d(grlp)}
json.dump(res, open("shapes.json", "w"), indent=1)
print({k: len(v) for k, v in res.items()})
