import re
import pyproj
from fastkml import kml, Placemark
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import transform
from shapely.affinity import rotate
import xml.etree.ElementTree as ET


def debug_print(*args):
    print("[DEBUG]", *args)


def fetch_kml(url_or_path):
    if url_or_path.startswith("http"):
        import requests
        response = requests.get(url_or_path)
        response.raise_for_status()
        return response.text
    else:
        with open(url_or_path, "r", encoding="utf-8") as file:
            return file.read()


def parse_polygon_from_kml(kml_text):
    debug_print("Parsing KML polygon...")

    if kml_text.lstrip().startswith("<?xml"):
        kml_text = kml_text[kml_text.find("?>") + 2:]

    kml_text = re.sub(r'<kml [^>]*>', '<kml xmlns="http://www.opengis.net/kml/2.2">', kml_text, count=1)
    kml_text = re.sub(r'kml:', '', kml_text)
    kml_text = re.sub(r'gx:', '', kml_text)

    k = kml.KML()
    k.from_string(kml_text.encode("utf-8"))

    features = list(k.features)
    if not features:
        root = ET.fromstring(kml_text)
        coords_text = None
        for elem in root.iter():
            if elem.tag.endswith('coordinates'):
                coords_text = elem.text
                break
        if coords_text:
            coord_pairs = [tuple(map(float, coord.split(',')[:2])) for coord in coords_text.strip().split()]
            if len(coord_pairs) >= 3:
                return Polygon(coord_pairs)
            else:
                raise ValueError("Not enough coordinates for a polygon.")
        raise ValueError("No features found in KML file.")

    document = features[0]
    doc_features = list(document.features)
    if not doc_features:
        raise ValueError("Document has no features")

    def find_coordinates(element):
        if hasattr(element, "coordinates"):
            coords = element.coordinates
            if coords and coords.strip():
                coord_pairs = [tuple(map(float, coord.split(',')[:2])) for coord in coords.strip().split()]
                if len(coord_pairs) < 3:
                    return None
                return coord_pairs
        if hasattr(element, "geometry"):
            return find_coordinates(element.geometry)
        if hasattr(element, "exterior"):
            return find_coordinates(element.exterior)
        return None

    for feature in doc_features:
        if isinstance(feature, Placemark):
            coords = find_coordinates(feature)
            if coords:
                return Polygon(coords)

    raise ValueError("No valid polygon coordinates found in KML file.")


def get_utm_transformers(lat, lon):
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "south" if lat < 0 else ""
    proj_latlon = pyproj.CRS("EPSG:4326")
    proj_utm = pyproj.CRS.from_proj4(
        f"+proj=utm +zone={utm_zone} {hemisphere} +datum=WGS84 +units=m +no_defs"
    )
    transformer_fwd = pyproj.Transformer.from_crs(proj_latlon, proj_utm, always_xy=True)
    transformer_back = pyproj.Transformer.from_crs(proj_utm, proj_latlon, always_xy=True)
    return transformer_fwd, transformer_back


def generate_lawnmower_path(polygon, spacing=10, altitude=5, grid_angle=0):
    centroid = polygon.centroid
    transformer_fwd, transformer_back = get_utm_transformers(centroid.y, centroid.x)
    polygon_utm = transform(transformer_fwd.transform, polygon)

    if grid_angle != 0:
        polygon_utm = rotate(polygon_utm, grid_angle, origin='centroid', use_radians=False)

    minx, miny, maxx, maxy = polygon_utm.bounds
    y = miny
    direction = 1
    waypoints = []

    while y <= maxy:
        sweep_line = LineString([(minx, y), (maxx, y)])
        clipped = sweep_line.intersection(polygon_utm)

        if clipped.is_empty:
            y += spacing
            continue

        segments = []
        if isinstance(clipped, LineString):
            segments = [clipped]
        elif isinstance(clipped, MultiLineString):
            segments = list(clipped.geoms)

        for segment in segments:
            coords = list(segment.coords)
            if direction == -1:
                coords.reverse()
            for x, y_coord in coords:
                lon, lat = transformer_back.transform(x, y_coord)
                waypoints.append({
                    "latitude": round(lat, 7),
                    "longitude": round(lon, 7),
                    "altitude": altitude
                })

        direction *= -1
        y += spacing

    return waypoints


def save_as_mission_waypoints(waypoints, filename="mission.waypoints"):
    header = "QGC WPL 110\n"
    lines = [header]
    index = 0

    # Takeoff command at first waypoint
    lines.append(f"{index}\t1\t3\t22\t0\t0\t0\t0\t{waypoints[0]['latitude']}\t{waypoints[0]['longitude']}\t{waypoints[0]['altitude']}\t1\n")
    index += 1

    # Survey waypoints
    for wp in waypoints:
        lines.append(f"{index}\t0\t3\t16\t0\t0\t0\t0\t{wp['latitude']}\t{wp['longitude']}\t{wp['altitude']}\t1\n")
        index += 1

    # Return to Launch
    lines.append(f"{index}\t0\t3\t20\t0\t0\t0\t0\t0\t0\t0\t1\n")

    with open(filename, "w") as f:
        f.writelines(lines)

    print(f"[✔] Waypoints saved to: {filename}")


def generate_waypoints_from_kml(kml_path, spacing=10, altitude=5, grid_angle=0):
    kml_text = fetch_kml(kml_path)
    polygon = parse_polygon_from_kml(kml_text)
    return generate_lawnmower_path(polygon, spacing, altitude, grid_angle)


def calculate_spacing(polygon, desired_lines=15, min_spacing=1, max_spacing=10):
    # Get polygon size in meters in UTM
    centroid = polygon.centroid
    transformer_fwd, _ = get_utm_transformers(centroid.y, centroid.x)
    polygon_utm = transform(transformer_fwd.transform, polygon)
    minx, miny, maxx, maxy = polygon_utm.bounds
    width = maxx - minx

    # Calculate spacing based on desired lines
    spacing = width / desired_lines

    # Clamp spacing within min/max limits
    spacing = max(min_spacing, min(spacing, max_spacing))
    print(f"[DEBUG] Auto-calculated spacing: {spacing:.2f} meters")
    return spacing


# --- MAIN ---
if __name__ == "__main__":
    kml_path = "/home/sherlock/Downloads/ground.kml"  # Your KML file path
    output_file = "/home/sherlock/Downloads/lawnmower_path.waypoints"  # Output file path

    kml_text = fetch_kml(kml_path)
    polygon = parse_polygon_from_kml(kml_text)

    spacing = calculate_spacing(polygon)  # Auto spacing based on polygon size
    altitude = 10  # Flight altitude in meters
    grid_angle = 0  # Rotation angle in degrees

    waypoints = generate_lawnmower_path(polygon, spacing, altitude, grid_angle)
    save_as_mission_waypoints(waypoints, output_file)
