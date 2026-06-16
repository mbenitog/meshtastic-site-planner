<template>
  <div>
    <p class="mt-hint mb-3">Statistical confidence and the maximum range to compute. Longer ranges take longer.</p>
    <div class="grid grid-cols-2 gap-2">
      <div>
        <label for="situation_fraction" class="mt-label">Situation Fraction (%)</label>
        <input v-model="simulation.situation_fraction" type="number" class="mt-input" id="situation_fraction" min="1" max="100" step="0.1" />
      </div>
      <div>
        <label for="time_fraction" class="mt-label">Time Fraction (%)</label>
        <input v-model="simulation.time_fraction" type="number" class="mt-input" id="time_fraction" min="1" max="100" step="0.1" />
      </div>
      <div class="col-span-2">
        <label for="simulation_extent" class="mt-label">Max Range (km)</label>
        <input v-model="simulation.simulation_extent" type="number" class="mt-input" id="simulation_extent" :min="simulation.ultra_backend ? 0.02 : 1" :max="simulation.ultra_backend ? 0.25 : (simulation.high_resolution ? 70 : 150)" :step="simulation.ultra_backend ? 0.01 : 1" />
      </div>
    </div>

    <label class="mt-3 flex cursor-pointer items-center gap-3">
      <span class="relative inline-flex shrink-0">
        <input v-model="simulation.high_resolution" type="checkbox" class="peer sr-only" id="high_resolution" />
        <span class="block h-5 w-9 rounded-full bg-surface-3 transition-colors peer-checked:bg-primary"></span>
        <span class="absolute top-0.5 left-0.5 size-4 rounded-full bg-ink transition-transform peer-checked:translate-x-4"></span>
      </span>
      <span class="text-sm font-medium text-ink">High resolution terrain (30 m)</span>
    </label>
    <p class="mt-hint mt-1">9x more detail than the default 90 m grid. Slower and limited to a 70 km range.</p>

    <label class="mt-3 flex cursor-pointer items-center gap-3">
      <span class="relative inline-flex shrink-0">
        <input v-model="simulation.ultra_backend" type="checkbox" class="peer sr-only" id="ultra_backend" />
        <span class="block h-5 w-9 rounded-full bg-surface-3 transition-colors peer-checked:bg-primary"></span>
        <span class="absolute top-0.5 left-0.5 size-4 rounded-full bg-ink transition-transform peer-checked:translate-x-4"></span>
      </span>
      <span class="text-sm font-medium text-ink">Ultra backend prototype (2.5 m)</span>
    </label>
    <p class="mt-hint mt-1">Uses the FastAPI backend, measured IGN DSM artifacts, and projected-grid ITM. Direct synchronous runs are currently limited to 250 m radius until tiled execution lands.</p>

    <div v-if="simulation.ultra_backend" class="mt-3">
      <label for="ultra_backend_url" class="mt-label">Ultra backend URL</label>
      <input v-model="simulation.ultra_backend_url" type="url" class="mt-input" id="ultra_backend_url" placeholder="http://127.0.0.1:8000" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { watch } from 'vue';
import { useStore } from '../store.ts';
const simulation = useStore().splatParams.simulation;

watch(
  () => simulation.ultra_backend,
  (enabled) => {
    if (enabled && simulation.simulation_extent > 0.25) simulation.simulation_extent = 0.25;
    if (!enabled && simulation.simulation_extent < 1) simulation.simulation_extent = 1;
  }
);
</script>
