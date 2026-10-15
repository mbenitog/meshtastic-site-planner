# Variables de Entorno del Sistema (Ultra Backend, Engine GPU y Frontend)

Este documento detalla todas las variables de entorno configurables en el proyecto, clasificadas por subsistema, indicando sus valores por defecto, comportamiento y casos de uso recomendados.

---

## 1. Fase 1: Descarga y Construcción de Superficie IGN (DTM / DSM)

Variables que controlan la interacción con las APIs del Instituto Geográfico Nacional (IGN) de España y el timeout del proceso de superficie.

| Variable | Valor por defecto | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| `ULTRA_SURFACE_TIMEOUT_S` | *Dinámico cuadrático* | Entero (segundos) | Límite máximo de tiempo para la descarga y procesado de la superficie 2.5 m (Fase 1). Si no se define, se calcula automáticamente según el área: `max(900, 600 + 450 * (radius_km / 10)^2)`. |
| `IGN_DOWNLOAD_CONCURRENCY` | `16` | Entero | Número de hilos simultáneos para la descarga paralela de subtelesas del IGN (WCS y Coverage API). Se puede elevar a `24` o `32` en conexiones de fibra de alta velocidad. |

### Ejemplo:
```bash
# Forzar 1 hora de timeout para simulaciones masivas (30 km - 50 km)
export ULTRA_SURFACE_TIMEOUT_S=3600
# Aumentar concurrencia de descarga
export IGN_DOWNLOAD_CONCURRENCY=24
```

---

## 2. Fase 2: Paralelización y Teselado de Cobertura RF

Variables que controlan el tamaño de las teselas y la concurrencia de workers en la simulación Longley-Rice / ITM.

| Variable | Valor por defecto | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| `ULTRA_TILE_MAX_CELLS` | *Auto-detectado* | Entero (número de celdas) | Celdas máximas procesadas por tesela. Si no se especifica: `2,000,000` en GPUs de $\ge 6\text{ GB}$; `1,000,000` en GPUs de $4\text{ GB}$ o ejecución en CPU. |
| `ULTRA_TILE_WORKERS` | *Auto-detectado* | Entero | Número de procesos concurrentes procesando teselas simultáneamente. En GPU se calcula asignando el 85% de la VRAM libre entre ~2.2 GB por worker. En CPU usa `max(1, cpu_count // 4)`. |
| `ULTRA_CLI` | `engine/build/ultra_cli` | Ruta (fichero ejecutable) | Ruta al binario nativo del motor de propagación ITM (`ultra_cli` para CPU o `ultra_cli_opencl` para GPU). En Docker se fija a `/usr/local/bin/ultra_cli_opencl`. |
| `ULTRA_OPENCL_KERNEL` | `""` | Ruta (fichero `.cl`) | Ruta al kernel OpenCL (`itm_kernel.cl`). Si está presente y el binario soporta OpenCL, se activa automáticamente la aceleración GPU. En Docker se fija a `/usr/local/share/ultra/itm_kernel.cl`. |

### Ejemplo:
```bash
# Forzar a 4 workers simultáneos y teselas de 2M de celdas
export ULTRA_TILE_WORKERS=4
export ULTRA_TILE_MAX_CELLS=2000000
```

---

## 3. Motor OpenCL: Gestión de VRAM y Selección de Dispositivo GPU

Variables evaluadas directamente por el binario C++ OpenCL (`ultra_cli_opencl`):

| Variable | Valor por defecto | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| `ULTRA_OPENCL_DEVICES` | *Auto-detección inteligente* | Cadena (separada por comas) | **Selección Multi-GPU**. Permite especificar qué dispositivos OpenCL utilizar en paralelo. Por defecto filtra automáticamente CPUs y GPUs integradas si hay tarjetas dedicadas. Admite índices (`0,1`), nombres/subcadenas (`"RTX 2070, RTX 3060"` o `"NVIDIA"`), o `"all_gpu"`. |
| `ULTRA_OPENCL_DEVICE` | *Auto-detectado* | Entero o Cadena | Dispositivo único OpenCL a seleccionar (ej. `0` o `1`). Si se define, fuerza el uso exclusivo de ese dispositivo. |
| `ULTRA_OPENCL_PLATFORM` | *Primera compatible* | Cadena o Entero | Nombre o índice de la plataforma OpenCL a utilizar (ej. `NVIDIA CUDA`, `0`). |
| `ULTRA_VRAM_FRACTION` | `0.85` | Float (`0.1` a `0.95`) | Fracción de la VRAM disponible que el motor OpenCL puede utilizar para reservar buffers de rayos y perfiles de elevación (0.85 = 85%). |
| `ULTRA_MEM_BUDGET_MB` | *Auto-calculado por GPU/worker* | Entero (Megabytes) | Presupuesto máximo estricto de memoria en MB para los buffers de rayos y perfiles por cada worker. |

### Ejemplos Multi-GPU:
```bash
# Usar dos GPUs dedicadas en paralelo (reparto round-robin de teselas)
export ULTRA_OPENCL_DEVICES="0,1"

# Filtrar por marca/nombre evitando GPUs integradas de Intel
export ULTRA_OPENCL_DEVICES="NVIDIA"

# Usar exclusivamente la segunda GPU
export ULTRA_OPENCL_DEVICES="1"
```

---

## 4. API Backend FastAPI / Servicio Ultra

Variables para configurar el enrutamiento y exposición de la API de Ultra:

| Variable | Valor por defecto | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| `ULTRA_API_PREFIX` | `/ultra-api` | Cadena (ruta URL) | Prefijo base para todas las rutas HTTP de la API de Ultra (`/jobs`, `/coverage/probe`, etc.). |
| `ULTRA_PUBLIC_BASE_URL` | `""` | Cadena (URL absoluta) | URL base pública utilizada para construir enlaces absolutos a los artefactos generados (útil detrás de reverse proxies como Traefik o Nginx). |
| `PYTHONPATH` | *No fijado localmente* | Cadena (rutas) | Ruta de búsqueda de módulos de Python (en Docker se establece en `/app` para permitir `from backend.ultra import ...`). |

---

## 5. Frontend / Aplicación Web

Variables evaluadas durante la compilación o ejecución del cliente web con Vite:

| Variable | Valor por defecto | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| `VITE_STADIA_API_KEY` | `""` | Cadena (clave de API) | Clave opcional de Stadia Maps para cargar la capa base de relieve "Terrain" (Stamen shaded-relief). No requerida en `localhost`. |

---

## Resumen de Configuración Rápida para Docker (`docker-compose.yml`)

Para exprimir al máximo una tarjeta GPU (ej. RTX 2070 / RTX 3060 de 8GB a 12GB):

```yaml
services:
  ultra-backend:
    image: meshtastic-ultra:latest
    environment:
      # Render GPU OpenCL
      - ULTRA_CLI=/usr/local/bin/ultra_cli_opencl
      - ULTRA_OPENCL_KERNEL=/usr/local/share/ultra/itm_kernel.cl
      - ULTRA_VRAM_FRACTION=0.85

      # Ajustes de paralelizado
      # (Dejar sin definir para auto-escalado automático según VRAM libre)
      # - ULTRA_TILE_WORKERS=3
      # - ULTRA_TILE_MAX_CELLS=2000000

      # Descargas Fase 1 IGN
      - IGN_DOWNLOAD_CONCURRENCY=20
      # - ULTRA_SURFACE_TIMEOUT_S=2400

      # API Prefix
      - ULTRA_API_PREFIX=/ultra-api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
