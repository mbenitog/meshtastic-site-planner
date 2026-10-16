<template>
  <div>
    <p class="mt-hint mb-3">The radio whose coverage is simulated.</p>
    <div class="grid grid-cols-2 gap-2">
      <div class="col-span-2">
        <label for="name" class="mt-label">Site name</label>
        <input v-model="transmitter.name" class="mt-input" id="name" title="Site name" />
      </div>
      <div>
        <label for="tx_lat" class="mt-label">Latitude (degrees)</label>
        <input v-model="transmitter.tx_lat" type="number" class="mt-input" id="tx_lat" min="-90" max="90" step="0.000001" title="Transmitter latitude in degrees (-90 to 90)." />
      </div>
      <div>
        <label for="tx_lon" class="mt-label">Longitude (degrees)</label>
        <input v-model="transmitter.tx_lon" type="number" class="mt-input" id="tx_lon" min="-180" max="180" step="0.000001" title="Transmitter longitude in degrees (-180 to 180)." />
      </div>
      <!-- Real-time elevation probe (Option B) -->
      <div class="col-span-2 my-1">
        <div v-if="store.elevationProbeLoading" class="p-2 rounded bg-slate-800/80 border border-slate-700/60 text-xs flex items-center gap-2 text-slate-300">
          <span class="mt-spinner" aria-hidden="true"></span>
          <span>Midiendo cota del terreno y superficie (IGN LiDAR)…</span>
        </div>
        <div v-else-if="store.elevationProbe && store.elevationProbe.available" class="p-2.5 rounded bg-slate-800/90 border border-slate-700 text-xs flex flex-col gap-1.5 shadow-sm">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-1.5">
              <span v-if="(store.elevationProbe.structure_m ?? 0) > 0.5" class="text-emerald-400 font-semibold flex items-center gap-1">
                🏢 Superficie: {{ store.elevationProbe.surface_m }} m
                <span class="text-emerald-300/80 font-normal text-[11px]">(+{{ store.elevationProbe.structure_m }} m estructura)</span>
              </span>
              <span v-else class="text-slate-200 font-semibold flex items-center gap-1">
                📍 Cota suelo: {{ store.elevationProbe.surface_m }} m s.n.m.
              </span>
            </div>
            <span class="text-[10px] text-slate-400 bg-slate-900/60 px-1.5 py-0.5 rounded border border-slate-700/50">
              {{ store.elevationProbe.source }}
            </span>
          </div>
          <div class="text-[11px] text-slate-400 flex justify-between">
            <span>Terreno base (DTM): <strong>{{ store.elevationProbe.ground_m }} m</strong></span>
            <span>Total con antena: <strong class="text-slate-200">{{ ((store.elevationProbe.surface_m ?? 0) + (transmitter.tx_height || 0)).toFixed(1) }} m ASL</strong></span>
          </div>
          <!-- Nearby high structure alert -->
          <div v-if="store.elevationProbe.nearby_structure" class="mt-1 pt-1.5 border-t border-slate-700/60 flex items-center justify-between gap-2 text-amber-300 bg-amber-950/30 -mx-1 px-2 py-1 rounded">
            <span class="text-[11px] leading-tight">
              ⚠️ Estructura alta a <strong>{{ store.elevationProbe.nearby_structure.distance_m }} m</strong> (cota <strong>{{ store.elevationProbe.nearby_structure.elevation_m }} m</strong>, +{{ store.elevationProbe.nearby_structure.height_above_point_m }} m)
            </span>
            <button
              type="button"
              @click="snapToNearbyStructure"
              class="mt-btn mt-btn-sm py-0.5 px-2 text-[10px] bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold whitespace-nowrap cursor-pointer"
              title="Mover el transmisor a la azotea detectada"
            >
              Mover a azotea
            </button>
          </div>
        </div>
        <div v-else-if="store.elevationProbe && !store.elevationProbe.available" class="p-2 rounded bg-slate-800/80 border border-slate-700/60 text-xs flex items-center justify-between gap-2 text-slate-400">
          <span>{{ store.elevationProbe.error || 'Cota no disponible o fuera de cobertura IGN' }}</span>
          <button
            type="button"
            @click="store.fetchElevationProbe(Number(transmitter.tx_lat), Number(transmitter.tx_lon))"
            class="mt-btn mt-btn-sm py-0.5 px-2 text-[10px] whitespace-nowrap"
          >
            Reintentar
          </button>
        </div>
      </div>
      <div class="col-span-2">
        <label for="device" class="mt-label">Device (optional)</label>
        <select v-model="selectedDevice" @change="applyDevice" class="mt-select" id="device" title="Prefill power and stock-antenna gain from a common Meshtastic device.">
          <option value="">Custom / manual</option>
          <option v-for="(d, i) in DEVICE_PROFILES" :key="i" :value="i">{{ d.label }}</option>
        </select>
        <p class="mt-hint mt-1">Fills typical power &amp; stock-antenna gain. Tune for your antenna and region.</p>
      </div>
      <div>
        <label for="tx_power" class="mt-label">Power (W)</label>
        <input v-model="transmitter.tx_power" type="number" class="mt-input" id="tx_power" min="0" step="0.1" title="Transmitter power in watts (>0)." />
      </div>
      <div>
        <label for="tx_freq" class="mt-label">Frequency (MHz)</label>
        <input v-model="transmitter.tx_freq" type="number" class="mt-input" id="tx_freq" min="20" max="20000" step="0.1" title="Transmitter frequency in MHz (20 to 20,000)." />
      </div>
      <div>
        <label for="tx_height" class="mt-label">Height AGL (m)</label>
        <input v-model="transmitter.tx_height" type="number" class="mt-input" id="tx_height" min="1.0" step="0.1" title="Transmitter height above ground in meters (>= 1.0)." />
      </div>
      <div>
        <label for="tx_gain" class="mt-label">Antenna Gain (dBi)</label>
        <input v-model="transmitter.tx_gain" type="number" class="mt-input" id="tx_gain" min="0" step="0.1" />
      </div>
    </div>
    <div class="mt-3 flex gap-2">
      <button
        @click="store.beginPlaceOnMap()"
        type="button"
        id="setWithMap"
        class="mt-btn mt-btn-sm flex-1 whitespace-nowrap"
        :class="store.placingMode ? 'mt-btn-secondary' : 'mt-btn-primary'"
      >
        {{ store.placingMode ? 'Click the map…' : 'Place on map' }}
      </button>
      <button @click="centerMapOnTransmitter" type="button" class="mt-btn mt-btn-secondary mt-btn-sm flex-1 whitespace-nowrap">
        Center on site
      </button>
    </div>
    <div class="mt-2 flex items-end gap-2">
      <div class="w-24">
        <label for="hp_radius" class="mt-label">Search km</label>
        <input id="hp_radius" v-model.number="hpRadius" type="number" min="0.2" max="10" step="0.5" class="mt-input" title="Radius to search for higher ground." />
      </div>
      <button
        type="button"
        class="mt-btn mt-btn-secondary mt-btn-sm flex-1 whitespace-nowrap"
        :disabled="store.highpointBusy"
        title="Move the transmitter to the highest ground nearby."
        @click="store.findHighpoint(hpRadius)"
      >
        <span v-if="store.highpointBusy" class="mt-spinner" aria-hidden="true"></span>
        {{ store.highpointBusy ? 'Searching…' : 'Snap to highest point' }}
      </button>
    </div>
    <p v-if="store.highpointMessage" class="mt-hint mt-1">{{ store.highpointMessage }}</p>
    <p class="mt-hint mt-2">Tip: drag the green pin to fine-tune, or type coordinates above.</p>
  </div>
</template>

<script setup lang="ts">
import { useStore } from '../store.ts';
import { onMounted, watch, ref } from 'vue';
import { DEVICE_PROFILES } from '../deviceProfiles';
const store = useStore();
const transmitter = store.splatParams.transmitter;

// Find-highpoint (#39) search radius, km.
const hpRadius = ref(1);

// Optional device quick-fill (#51): selecting a radio prefills power + gain.
const selectedDevice = ref<number | ''>('');
function applyDevice() {
  if (selectedDevice.value === '') return;
  const d = DEVICE_PROFILES[selectedDevice.value as number];
  if (!d) return;
  transmitter.tx_power = d.tx_power;
  transmitter.tx_gain = d.tx_gain;
}

// If power or gain is hand-edited away from the chosen device, fall back to
// "Custom" so the dropdown never misrepresents the current values.
watch(
  () => [Number(transmitter.tx_power), Number(transmitter.tx_gain)] as const,
  ([p, g]) => {
    if (selectedDevice.value === '') return;
    const d = DEVICE_PROFILES[selectedDevice.value as number];
    if (d && (p !== d.tx_power || g !== d.tx_gain)) selectedDevice.value = '';
  }
);

const centerMapOnTransmitter = () => {
  if (!isNaN(transmitter.tx_lat) && !isNaN(transmitter.tx_lon)) {
    store.getMap()?.flyTo({ center: [transmitter.tx_lon, transmitter.tx_lat] });
  } else {
    alert('Please enter valid Latitude and Longitude values.');
  }
};

function snapToNearbyStructure() {
  const nearby = store.elevationProbe?.nearby_structure;
  if (!nearby) return;
  transmitter.tx_lat = nearby.lat;
  transmitter.tx_lon = nearby.lon;
  store.setTxCoords(nearby.lat, nearby.lon);
  store.setDraftMarker(nearby.lat, nearby.lon);
  store.getMap()?.flyTo({ center: [nearby.lon, nearby.lat], zoom: Math.max(store.getMap()?.getZoom() ?? 15, 16) });
}

// Typing coordinates moves (or creates) the draggable draft pin, so the
// text fields and the map marker always agree. Skipped mid-simulation.
watch(
  () => [Number(transmitter.tx_lat), Number(transmitter.tx_lon)] as const,
  ([lat, lon]) => {
    if (store.simulationState === 'running') return;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;
    store.setDraftMarker(lat, lon);
    store.fetchElevationProbeDebounced(lat, lon);
  }
);

onMounted(() => {
  store.initMap(); // Initialize the map
  const lat = Number(transmitter.tx_lat);
  const lon = Number(transmitter.tx_lon);
  if (Number.isFinite(lat) && Number.isFinite(lon)) {
    store.fetchElevationProbe(lat, lon);
  }
});
</script>
