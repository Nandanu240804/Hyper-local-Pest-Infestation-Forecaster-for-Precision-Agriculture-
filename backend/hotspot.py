# backend/hotspot.py
import math

def generate_hotspots(lat, lon, confidence, n=2, base_radius_m=500):
    """
    Simple hotspot generator:
      - returns n hotspots around (lat, lon) with offsets determined by confidence.
      - base_radius_m is the base radius in meters; final radius scales with confidence.
    NOTE: This uses a naive lat/lon meter -> degree conversion (approx).
    """
    if lat is None or lon is None:
        lat, lon = 12.9716, 77.5946

    # scale radius by confidence (confidence in [0,1])
    c = float(confidence) if confidence is not None else 0.5
    radius = base_radius_m * (0.5 + c)  # between 0.5*base .. 1.5*base

    # approximate conversion: 1 deg lat ~ 111 km; 1 deg lon ~ 111km*cos(lat)
    deg_per_m_lat = 1.0 / 111000.0
    deg_per_m_lon = 1.0 / (111000.0 * math.cos(math.radians(lat)))

    # create n hotspots around the center at various bearings
    hotspots = []
    for i in range(n):
        bearing = (i * 360 / n) + 30  # degrees
        # simple offset
        dx = radius * math.cos(math.radians(bearing))
        dy = radius * math.sin(math.radians(bearing))
        new_lat = lat + (dy * deg_per_m_lat)
        new_lon = lon + (dx * deg_per_m_lon)
        hotspots.append({"lat": new_lat, "lon": new_lon, "radius_m": radius})

    return hotspots

# backend/hotspot.py
import math
import random

def move_lat_lon(lat, lon, bearing_deg, meters):
    """
    Move (lat, lon) by `meters` along `bearing_deg` and return new lat/lon.
    Uses haversine-based forward formula (sufficient for small distances).
    """
    R = 6378137.0  # Earth radius in meters
    br = math.radians(bearing_deg)
    d = meters
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(d/R) + math.cos(lat1) * math.sin(d/R) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(d/R) * math.cos(lat1),
                             math.cos(d/R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def generate_hotspots(lat, lon, confidence=0.5, hours=24, n=4,
                      base_speed_m_per_hr=2.0, alpha=1.0,
                      radius_factor=0.6, jitter=0.15, seed=None):
    """
    Generate simple hotspot points around (lat, lon).
    Parameters:
      - confidence: model confidence in [0,1]
      - hours: forecast horizon in hours (24,48,72)
      - n: number of hotspots (4 is typical: N,S,E,W or quadrants)
      - base_speed_m_per_hr: expected mean spread speed in meters/hour (default 2 m/hr)
      - alpha: confidence multiplier (how much confidence increases spread)
      - radius_factor: fraction of distance used as circle radius (0.3-1.0)
      - jitter: random variation (0..0.5); reduce to 0.0 for deterministic
      - seed: optional RNG seed for reproducible results
    Returns:
      list of dicts: {"lat":..,"lon":..,"radius_m":..,"score":..}
    """

    if seed is not None:
        random.seed(seed)

    # clamp inputs
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    hours = max(1, int(hours))
    n = max(1, int(n))

    # Compute mean spread distance (meters)
    # distance increases with time and slightly with confidence
    distance = base_speed_m_per_hr * hours * (1.0 + alpha * confidence)

    hotspots = []
    bearings = [i * (360.0 / n) for i in range(n)]

    for b in bearings:
        # small jitter to distance and bearing for realism
        d_jitter = distance * (1.0 + (random.uniform(-jitter, jitter)))
        bearing_jitter = b + random.uniform(-10.0, 10.0)  # +-10 degrees

        lat2, lon2 = move_lat_lon(lat, lon, bearing_jitter, d_jitter)

        # radius ~ fraction of distance (so hotspot area is smaller for small distances)
        radius = max(5.0, distance * radius_factor * (1.0 + random.uniform(-jitter, jitter)))

        # score scales with confidence and a small random factor
        score = float(max(0.0, min(1.0, confidence * (0.6 + 0.4 * random.random()))))

        hotspots.append({
            "lat": lat2,
            "lon": lon2,
            "radius_m": float(radius),
            "score": score,
            "bearing": bearing_jitter,
            "distance_m": float(d_jitter)
        })

    # sort by score descending (optional)
    hotspots.sort(key=lambda h: h["score"], reverse=True)
    return hotspots
