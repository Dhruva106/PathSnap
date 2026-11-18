/**
 * map.js - Advanced Leaflet Map Controller for PathSnap Geospatial Dashboard
 */

class PathSnapMap {
  constructor(containerId) {
    this.containerId = containerId;
    this.map = null;

    // Feature Groups for multi-layer rendering
    this.gpsLayer = L.featureGroup();
    this.snappedPointsLayer = L.featureGroup();
    this.reconstructedRouteLayer = L.featureGroup();
    this.groundTruthLayer = L.featureGroup();
    this.teammateLayer = L.featureGroup();
    this.highlightLayer = L.featureGroup();
    this.markerLayer = L.featureGroup();
    this.teammateLayers = {};

    this.currentBounds = [];
    this.initMap();
  }

  initMap() {
    // Default center (Beijing coordinates where dataset is collected: ~39.9, 116.4)
    this.map = L.map(this.containerId, {
      center: [39.9042, 116.4074],
      zoom: 13,
      zoomControl: true,
    });

    // Dark Matter tile layer for high-contrast geospatial analytics
    const darkTiles = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>, OpenStreetMap',
        maxZoom: 19,
        subdomains: "abcd",
      }
    );

    // Standard OpenStreetMap fallback
    const osmTiles = L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
      }
    );

    darkTiles.addTo(this.map);

    // Base Layer Control
    L.control.layers(
      { "Dark Matter": darkTiles, "OpenStreetMap": osmTiles },
      null,
      { position: "bottomright" }
    ).addTo(this.map);

    // Add all feature groups
    this.gpsLayer.addTo(this.map);
    this.reconstructedRouteLayer.addTo(this.map);
    this.snappedPointsLayer.addTo(this.map);
    this.groundTruthLayer.addTo(this.map);
    this.teammateLayer.addTo(this.map);
    this.markerLayer.addTo(this.map);
    this.highlightLayer.addTo(this.map);
  }

  clear() {
    this.gpsLayer.clearLayers();
    this.snappedPointsLayer.clearLayers();
    this.reconstructedRouteLayer.clearLayers();
    this.groundTruthLayer.clearLayers();
    this.teammateLayer.clearLayers();
    this.markerLayer.clearLayers();
    this.highlightLayer.clearLayers();
    this.clearTeammateRoutes();
    this.currentBounds = [];
  }

  renderTeammateRoute(implName, color, coords, metrics = {}) {
    if (!coords || coords.length === 0) return;

    if (!this.teammateLayers[implName]) {
      this.teammateLayers[implName] = L.featureGroup().addTo(this.map);
    }
    const layer = this.teammateLayers[implName];
    layer.clearLayers();

    const latLngs = coords.map((c) => [c[0], c[1]]);
    latLngs.forEach((ll) => this.currentBounds.push(ll));

    const polyline = L.polyline(latLngs, {
      color: color || "#f59e0b",
      weight: 4,
      opacity: 0.85,
      dashArray: "5, 5",
      lineCap: "round",
    });

    const tip =
      `<div style="font-family: sans-serif; font-size: 11px;">` +
      `<strong style="color: ${color || "#f59e0b"};">${implName}</strong><br>` +
      `Accuracy: ${metrics.accuracy != null ? metrics.accuracy.toFixed(1) + "%" : "N/A"}<br>` +
      `F1-Score: ${metrics.f1_score != null ? metrics.f1_score.toFixed(1) + "%" : "N/A"}<br>` +
      `Continuity: ${metrics.route_continuity != null ? metrics.route_continuity.toFixed(1) + "%" : "N/A"}` +
      `</div>`;

    polyline.bindTooltip(tip, { sticky: true });
    layer.addLayer(polyline);
    this.teammateLayer.addLayer(polyline);
  }

  clearTeammateRoutes() {
    this.teammateLayer.clearLayers();
    Object.values(this.teammateLayers).forEach((layer) => {
      layer.clearLayers();
    });
    this.teammateLayers = {};
  }

  renderResults(data) {
    this.clear();
    const bounds = [];
    const matchedPts = data.matched_points || [];
    const routeCoords = data.reconstructed_route?.route_polyline_wgs84 || [];

    // 1. Draw Raw GPS Polyline and Point Markers
    const gpsLatLngs = [];
    matchedPts.forEach((pt) => {
      const latLng = [pt.original_latitude, pt.original_longitude];
      gpsLatLngs.push(latLng);
      bounds.push(latLng);

      const circle = L.circleMarker(latLng, {
        radius: 3,
        color: "#ef4444",
        fillColor: "#ef4444",
        fillOpacity: 0.7,
        weight: 1,
      });

      circle.bindTooltip(
        `<div style="font-family: sans-serif; font-size: 11px;">` +
        `<b>GPS Point #${pt.point_idx}</b><br>` +
        `Lat: ${pt.original_latitude.toFixed(6)}<br>` +
        `Lon: ${pt.original_longitude.toFixed(6)}<br>` +
        `Time: ${pt.timestamp.toFixed(1)}s` +
        `</div>`,
        { direction: "top" }
      );
      this.gpsLayer.addLayer(circle);
    });

    if (gpsLatLngs.length > 1) {
      const gpsLine = L.polyline(gpsLatLngs, {
        color: "#f87171",
        weight: 2,
        dashArray: "4, 6",
        opacity: 0.75,
      });
      this.gpsLayer.addLayer(gpsLine);
    }

    // 2. Draw Reconstructed Route Polyline (Bright Green)
    if (routeCoords.length > 0) {
      const routeLatLngs = routeCoords.map((coord) => [coord[0], coord[1]]);
      routeLatLngs.forEach((ll) => bounds.push(ll));

      const routePolyline = L.polyline(routeLatLngs, {
        color: "#10b981",
        weight: 5,
        opacity: 0.92,
        lineCap: "round",
        lineJoin: "round",
      });

      routePolyline.bindTooltip(
        `<div style="font-family: sans-serif; font-size: 12px;">` +
        `<b>Reconstructed Route</b><br>` +
        `Length: ${(data.reconstructed_route.total_length_meters / 1000).toFixed(2)} km<br>` +
        `Arcs: ${data.reconstructed_route.num_arcs}<br>` +
        `Continuity: ${data.reconstructed_route.is_fully_connected ? "100% Connected" : "Has Jumps"}` +
        `</div>`,
        { sticky: true }
      );
      this.reconstructedRouteLayer.addLayer(routePolyline);

      // Add Start and End Markers
      this.renderStartEndMarkers(routeLatLngs[0], routeLatLngs[routeLatLngs.length - 1]);
    }

    // 3. Draw Snapped Points (Cyan) and Tether Connectors
    matchedPts.forEach((pt) => {
      const matchedLatLng = [pt.matched_latitude, pt.matched_longitude];
      bounds.push(matchedLatLng);

      const snapMarker = L.circleMarker(matchedLatLng, {
        radius: 4,
        color: "#38bdf8",
        fillColor: "#0284c7",
        fillOpacity: 0.9,
        weight: 1.5,
      });

      snapMarker.bindPopup(
        `<div style="font-family: sans-serif; font-size: 12px; color: #1e293b; min-width: 170px;">` +
        `<strong style="color: #0284c7;">Observation #${pt.point_idx}</strong><br>` +
        `<b>Matched Road Arc:</b> ${pt.matched_arc_id}<br>` +
        `<b>ML Confidence:</b> ${(pt.confidence * 100).toFixed(1)}%<br>` +
        `<b>Snapping Offset:</b> ${pt.orthogonal_distance_m.toFixed(2)} m<br>` +
        `<b>Candidates Evaluated:</b> ${pt.candidate_count}<br>` +
        `<b>Top-1 Argmax Arc:</b> ${pt.top1_ml_arc_id}` +
        `</div>`
      );
      this.snappedPointsLayer.addLayer(snapMarker);

      // Dashed connector tether from raw GPS to snapped point
      const connector = L.polyline(
        [[pt.original_latitude, pt.original_longitude], matchedLatLng],
        {
          color: "#94a3b8",
          weight: 1,
          opacity: 0.45,
          dashArray: "2, 3",
        }
      );
      this.snappedPointsLayer.addLayer(connector);
    });

    this.currentBounds = bounds;
    if (bounds.length > 0) {
      this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 17 });
    }
  }

  renderStartEndMarkers(startLatLng, endLatLng) {
    this.markerLayer.clearLayers();

    // Start Marker (Green Circle with Ring)
    const startIcon = L.divIcon({
      className: "custom-marker-start",
      html: `<div style="background: #10b981; width: 14px; height: 14px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 10px #10b981;"></div>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
    const startMarker = L.marker(startLatLng, { icon: startIcon });
    startMarker.bindTooltip("<b>Trip Origin (Start)</b>", { direction: "top" });
    this.markerLayer.addLayer(startMarker);

    // End Marker (Red Circle with Ring)
    const endIcon = L.divIcon({
      className: "custom-marker-end",
      html: `<div style="background: #ef4444; width: 14px; height: 14px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 10px #ef4444;"></div>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
    const endMarker = L.marker(endLatLng, { icon: endIcon });
    endMarker.bindTooltip("<b>Trip Destination (End)</b>", { direction: "top" });
    this.markerLayer.addLayer(endMarker);
  }

  renderGroundTruth(coords) {
    this.groundTruthLayer.clearLayers();
    if (!coords || coords.length === 0) return;

    const gtPolyline = L.polyline(coords, {
      color: "#a855f7", // Vibrant Purple
      weight: 4,
      dashArray: "6, 8",
      opacity: 0.85,
    });

    gtPolyline.bindTooltip("<b>Reference Ground Truth Route (.route)</b>", { sticky: true });
    this.groundTruthLayer.addLayer(gtPolyline);

    // Bring predicted route to front so comparison is clear
    this.reconstructedRouteLayer.bringToFront();
  }

  highlightPoint(lat, lon, matchedLat, matchedLon, ptIdx, arcId) {
    this.highlightLayer.clearLayers();

    // Pulsing highlight circle
    const halo = L.circleMarker([lat, lon], {
      radius: 12,
      color: "#f59e0b",
      fillColor: "#f59e0b",
      fillOpacity: 0.35,
      weight: 2,
    });

    const tether = L.polyline([[lat, lon], [matchedLat, matchedLon]], {
      color: "#f59e0b",
      weight: 2.5,
      dashArray: "3, 4",
    });

    const snappedTarget = L.circleMarker([matchedLat, matchedLon], {
      radius: 6,
      color: "#f59e0b",
      fillColor: "#f59e0b",
      fillOpacity: 0.9,
      weight: 2,
    });

    this.highlightLayer.addLayer(halo);
    this.highlightLayer.addLayer(tether);
    this.highlightLayer.addLayer(snappedTarget);

    // Pan smoothly
    this.map.panTo([lat, lon], { animate: true, duration: 0.4 });
  }

  fitBounds() {
    if (this.currentBounds.length > 0) {
      this.map.fitBounds(this.currentBounds, { padding: [50, 50], maxZoom: 17 });
    }
  }

  toggleLayer(layerName, isVisible) {
    let layer = null;
    if (layerName === "gps") layer = this.gpsLayer;
    else if (layerName === "route") layer = this.reconstructedRouteLayer;
    else if (layerName === "snapped") layer = this.snappedPointsLayer;
    else if (layerName === "groundtruth") layer = this.groundTruthLayer;
    else if (layerName === "teammate") layer = this.teammateLayer;

    if (layer) {
      if (isVisible) {
        if (!this.map.hasLayer(layer)) this.map.addLayer(layer);
      } else {
        if (this.map.hasLayer(layer)) this.map.removeLayer(layer);
      }
    }
  }
}
