# orbitwatch walkthrough

A guided tour of the code, the physics and remote-sensing ideas behind it, the reasons
for each design decision, and the interview questions this project tends to invite.

---

## 1. The 60-second version

orbitwatch answers questions about two kinds of space data around Lompoc and Vandenberg
Space Force Base:

* **Orbits**: when is a satellite overhead, and can you see it? (CelesTrak TLEs + SGP4)
* **Earth observation**: how did vegetation (NDVI) or burn severity (NBR) change between
  two dates over a small area? (Sentinel-2 via the Earth Search STAC API)

Those capabilities are written **once**, as six typed methods on `Toolbox`
(`orbitwatch/tools.py`). Four front ends share them:

| Front end | File | Who calls the tools |
|---|---|---|
| CLI | `orbitwatch/cli.py` | you |
| MCP server | `orbitwatch/mcp_server.py` | Claude Code / Claude Desktop |
| Claude agent | `orbitwatch/agent/claude_agent.py` | Claude, via the Messages API |
| Offline planner | `orbitwatch/agent/planner.py` | a deterministic keyword planner (no LLM) |

Where tools get data is injected through `DataContext` (`orbitwatch/context.py`): **live**
(CelesTrak, Earth Search, JPL ephemeris) or **fixtures** (recorded metadata, synthetic
pixels, a fixed clock). Tests and CI evals run on fixtures with network access blocked.

---

## 2. Code tour (read in this order)

1. `orbitwatch/config.py` - ground sites (Lompoc, Vandenberg) with the source of each
   coordinate, the demo area of interest, defaults. Plain data.
2. `orbitwatch/orbits/tle.py` - parse and validate TLE text (fixed columns + mod-10
   checksum), epoch decoding.
3. `orbitwatch/orbits/celestrak.py` - fetch from CelesTrak with a 2-hour file cache, stale
   fallback, no retries on 403/404. `FileTleSource` is the offline twin.
4. `orbitwatch/orbits/passes.py` - Skyfield `find_events` -> grouped passes; visibility by
   sampling each pass for "satellite sunlit while observer's Sun is below -6 deg".
5. `orbitwatch/eo/stac.py` - STAC search, `Scene`/`BandAsset` dataclasses, and
   `effective_offset` (the reflectance-offset rule, see section 4.3).
6. `orbitwatch/eo/raster.py` - windowed COG reads onto one shared grid.
7. `orbitwatch/eo/indices.py` - pure numpy: DN -> reflectance, normalized differences, SCL mask.
8. `orbitwatch/eo/change.py` - change statistics and the PNG.
9. `orbitwatch/eo/analysis.py` - glue: scene -> windows -> index -> before/after summary.
10. `orbitwatch/tools.py` - the six tools, schema generation, validated dispatch.
11. `orbitwatch/agent/claude_agent.py` - the tool-use loop.
12. `orbitwatch/agent/planner.py` and `grounding.py` - offline planner and the
    "numbers must come from tools" checker.
13. `evals/` - golden questions, runner, brute-force pass reference.

---

## 3. Orbital basics in plain words

### TLE and SGP4

A **two-line element set (TLE)** is a compact snapshot of a satellite's orbit at one
instant, the **epoch**: inclination, eccentricity, mean motion (revolutions per day),
and a drag term (B*). The ISS TLE used in the tests has mean motion 15.488 rev/day, so one
orbit takes about 93 minutes, and inclination 51.63 deg.

TLE elements are *mean* elements fitted for one specific propagator, **SGP4**. SGP4
models the main perturbations for near-Earth orbits (Earth's oblateness, atmospheric drag
via B*, Sun/Moon effects in its deep-space branch). Feed a TLE to anything else and you get
the wrong orbit; that is why we use Skyfield's SGP4 implementation rather than Kepler's
equations.

**Accuracy decays with time from epoch**, fastest for low, draggy orbits like the ISS,
and the ISS also makes reboost manoeuvres a TLE cannot anticipate. So the tool reports the
TLE age and warns when the window runs more than 7 days past the epoch, and it caps the
window at 10 days.

### From orbit to "look up now"

For an observer at a latitude, longitude and height, Skyfield converts the satellite
position to **altitude** (elevation above the horizon) and **azimuth** (compass bearing).
A **pass** is the interval above a minimum elevation (10 deg by default; low passes are
blocked by terrain and haze). `find_events` returns rise, culmination (highest point) and
set; passes cut off by the window edges are dropped.

### Visibility

You can see a satellite with the naked eye when **it is in sunlight but your sky is
dark**: after sunset or before sunrise, the satellite is high enough to still catch the Sun.
orbitwatch uses the common rule "satellite sunlit and Sun below -6 deg" (civil twilight).
Both checks need the Sun's position, which comes from the JPL DE421 planetary ephemeris
(downloaded once, ~17 MB; tests use a 25 KB two-month excerpt).

### CelesTrak etiquette (and a 2026 wrinkle)

CelesTrak publishes the US Space Force's general perturbations (GP) data. Its
documentation says it checks for new data every 2 hours and enforces one download per
update cycle; repeated requests can get an IP firewalled. Hence the 2-hour cache, the
stale-cache fallback and no retry loop. It also notes the 5-digit catalog numbers ran out
on 2026-07-11: newer objects have no TLE at all, only OMM (JSON/XML/CSV) - see the roadmap.

---

## 4. Remote-sensing basics

### 4.1 Reflectance, L1C vs L2A

Sentinel-2's MultiSpectral Instrument measures sunlight reflected from the ground in 13
spectral bands at 10, 20 or 60 m pixels. **Level-2A** products are atmospherically
corrected to **bottom-of-atmosphere surface reflectance**: roughly, the fraction of
incoming light the surface reflects in each band (0 to 1). Pixels are stored as 16-bit
integers (DN); reflectance = DN x 0.0001 (+ an offset, see 4.3).

### 4.2 NDVI and NBR

* **NDVI = (NIR - Red) / (NIR + Red)**, bands B08 (842 nm) and B04 (665 nm), 10 m.
  Chlorophyll absorbs red light and leaf structure scatters near-infrared strongly, so
  dense healthy vegetation scores high (often 0.6-0.9); bare soil is low (~0.1-0.2); water
  is negative.
* **NBR = (NIR - SWIR2) / (NIR + SWIR2)**, bands B8A (865 nm) and B12 (2190 nm), 20 m.
  Burned or dry surfaces reflect more shortwave infrared and less NIR, so NBR drops.
  Fire analysts use **dNBR = pre-fire NBR - post-fire NBR** (positive = more severe);
  orbitwatch reports `after - before` for both indices, so a burn shows up as negative.

orbitwatch reports what these numbers do; it does not claim *why* they changed. A change
in NDVI between April and September could reflect crop cycles, irrigation, harvest,
seasonal drying of natural vegetation, or something else; telling those apart needs
ground truth or more data.

### 4.3 The offset trap (and how it was verified)

Since **processing baseline 04.00** (January 2022) ESA adds +1000 to every L2A DN so
dark pixels are not clipped at zero; the right conversion becomes DN x 0.0001 - 0.1.
Earth Search items declare that as `raster:bands: {scale: 0.0001, offset: -0.1}`.

But Earth Search also sets `earthsearch:boa_offset_applied: true` on these items, and
third-party reports said the pixels were already corrected. Rather than trust either
claim, `scripts/verify_boa_offset.py` compares the **same acquisition** (2021-06-15) that
Earth Search lists twice: as processed at baseline 03.00 (no offset) and reprocessed at
05.00 (flagged). Result of the run during this build:

| Area | baseline 03.00 red DN median | baseline 05.00 red DN median | difference |
|---|---|---|---|
| Pacific Ocean off Point Arguello | 433 | 424 | -9 |
| Lompoc Valley fields | 1288 | 1284 | -4 |

If the +1000 were still in the pixels the difference would be about +1000. So for flagged
items the offset is already removed, and applying the declared -0.1 would subtract it a
second time. On the worked-example scene (S2C_10SGD_20260420_0_L2A) the red-band median DN
over the AOI is 939; double-subtracting would make red reflectance negative in 53.09% of
pixels, leave only 46.91% valid, and move mean NDVI from 0.3789 to 0.4381. The rule lives in
one function (`effective_offset`) and every result reports which rule it applied.

There is a second, smaller numeric trap in the same place: in float32,
`1000 * 0.0001 - 0.1` is about `-7e-9`, which the negative-reflectance mask would discard.
`dn_to_reflectance` applies the offset in DN units first (`(DN - 1000) * 0.0001`), so a true
zero stays exactly zero. A unit test pins this.

### 4.4 Clouds: the Scene Classification Layer

Scene-level `eo:cloud_cover` describes a ~110 km tile, not your 3 km field. L2A ships a
20 m **Scene Classification Layer (SCL)**; orbitwatch masks no-data (0), saturated (1),
cloud shadow (3), medium/high-probability cloud (8, 9), thin cirrus (10) and snow (11), and
reports the valid fraction. SCL is a classifier and misses some thin cloud and haze, so
results carry a warning when less than half the pixels are clear in both scenes.

### 4.5 COGs and windowed reads

A **Cloud-Optimized GeoTIFF** stores pixels in internal tiles with the index at the start
of the file. GDAL reads the header, then fetches only the byte ranges for the tiles under
your window with HTTP range requests. A full 10 m Sentinel-2 band is 10980 x 10980 pixels;
the demo AOI is 344 x 378. The live NDVI comparison (six band windows across two scenes, plus
STAC lookups) took about 6 seconds on a home connection. The GDAL settings in
`raster.py` (`GDAL_DISABLE_READDIR_ON_OPEN`, merged range requests) avoid needless requests.

### 4.6 Lining up pixels

Bands come at 10 m and 20 m on the same UTM grid. The AOI is projected to the scene's UTM
zone and **snapped outward to the 20 m grid**, so every band's window covers the same
ground; 20 m bands are then read onto the 10 m grid with nearest-neighbour (each 20 m
pixel becomes a 2 x 2 block, no invented values). Before/after scenes must come from the
**same MGRS tile** (same CRS and grid); the tool refuses mismatched pairs instead of
silently resampling.

### 4.7 STAC

**STAC** is a JSON standard: a *catalog* holds *collections* (e.g. `sentinel-2-l2a`) of
*items* (one scene each: footprint, time, properties) whose *assets* are files (one COG per
band). A STAC API adds search by bbox, time and properties. Asset keys are not
standardised across providers, so the keys used here (`red`, `nir`, `nir08`, `swir22`,
`scl`) were read from live Earth Search responses, and a recorded response is a test
fixture.

---

## 5. Applied-AI design

### 5.1 Tools as the contract

Each tool is a `Toolbox` method with `Annotated[..., Field(description=...)]` parameters
and a docstring that says **what it does and when to call it** (Claude decides largely from
descriptions). Two consumers derive JSON Schema from the same signature:

* the MCP SDK (`MCPServer.add_tool`), and
* `tools.input_model()`, which builds a pydantic model with `extra="forbid"` for the
  Claude agent.

A test asserts both expose the same parameters and required fields, so they cannot drift.
Schemas are sent with `strict: true` and `additionalProperties: false`. Strict mode does
not support numeric bounds (`minimum`, `maxItems`, ...), so ranges are described in text
and enforced in code, with error messages written for the model to read and correct.

Tools return JSON with numbers **rounded at the source** (elevation to 0.1 deg, statistics
to 4 decimals), so the model can quote values exactly instead of re-rounding them.

### 5.2 The agent loop

`ClaudeAgent.ask` is a short manual loop over `client.beta.messages.create`:

1. Call with `model="claude-opus-5-5"`, `thinking={"type": "adaptive"}`,
   `output_config={"effort": "medium"}` (explicit; effort is the thinking control on this
   model), `tool_choice` left at auto, strict tools, and the server-side refusal fallback
   (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`).
2. Check `stop_reason` **before** touching tool calls: `refusal` stops (never run tools
   from a declined turn); `max_tokens` stops (a tool_use block may be truncated).
3. Append the full `response.content` (thinking blocks included, never edited), minus
   blocks before a `fallback` boundary, which belong to the declined attempt.
4. Run every `tool_use` block, return all results in **one** user message, `is_error: true`
   on failures so the model can recover.
5. Stop at `end_turn` or after 8 turns.

Why not the SDK's tool runner? It runs tools automatically. Here each turn must be
inspected before any tool executes, and every call is recorded for the evals. The loop is
a few dozen lines and each rule is visible in one place. One subtle requirement: the
system prompt (which includes the current time) is built **once per question**, because
the system prompt, tools and earlier messages must stay byte-identical across turns for
replayed thinking blocks to remain valid and for the prompt cache to hit; a unit test
pins this. The runner would be the right call
for an agent without those needs.

### 5.3 The offline planner, honestly labelled

Without an API key the project still has to be demoable and testable. `OfflinePlanner`
classifies a question with keyword rules, extracts parameters with regexes, calls the same
validated tools and fills a template **only with tool values**. Every answer starts with
`[OFFLINE MODE: scripted keyword planner, no LLM]`. Unsupported questions get a refusal,
never a guess.

### 5.4 Evals

`evals/golden.yaml` lists questions with checkable expectations: which tools must (and
must not) be called, with which arguments, plus:

* **grounded** - every number, date and time in the answer must appear in a tool result
  (`agent/grounding.py`; tolerant of rounding, percentages of fractions, AM/PM and local
  time, but nothing else);
* **cites_tle_epoch** / **cites_scenes** - the answer names the TLE epoch, or the scene
  IDs and acquisition dates it used.

Offline (CI) the runner drives the scripted planner on fixtures. That is a regression
test of the harness and tools, not a model eval: I wrote both the planner and the checks.
`--live` runs the identical cases through Claude, which is the real eval.

A separate **reference case** checks physics: the event-based pass finder must match an
independent brute-force 1-second elevation scan (`evals/make_pass_reference.py`) to 2 s and
0.05 deg. Both use Skyfield's SGP4, so this validates event finding, grouping and geometry,
not SGP4 itself.

### 5.5 Hermetic tests

`tests/conftest.py` blocks every non-loopback socket connection in non-live tests, so a
test that accidentally reaches the network fails loudly. Fixture scenes keep real recorded
STAC metadata but point at small synthetic GeoTIFFs generated on the fly; their IDs are
prefixed `FIXTURE-`, outputs say `FIXTURES (synthetic pixels)` and the PNG title says so,
so fixture numbers can never pass for measurements.

---

## 6. Decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| One `Toolbox`, schemas derived from signatures | MCP and agent cannot drift | Signature-to-model helper (`input_model`) to understand |
| Manual agent loop | Inspect stop_reason before tools; record calls | More code than the tool runner |
| Fixture mode via dependency injection | Same code path offline; hermetic CI | Fixture data to maintain |
| Synthetic pixels, real metadata | No imagery in git; realistic shapes | Fixture numbers are meaningless (and labelled so) |
| Skip declared offset when `boa_offset_applied` | Verified empirically on paired scenes | Depends on Earth Search keeping that convention |
| Mask negative reflectance | Avoids out-of-range indices | Very dark pixels are dropped |
| Same-tile pairs only | Exact pixel alignment, no resampling | Can't pair across UTM zones |
| 2-hour TLE cache, no retries | CelesTrak policy | Data up to 2 h old |
| TLE format | Asked for, simple, well known | No objects with 6-digit catalog numbers (needs OMM) |

---

## 7. Likely interview questions

**Why SGP4 and not a simple Keplerian propagator?**
TLEs are mean elements fitted for SGP4; using them elsewhere gives the wrong orbit. SGP4
also models drag and Earth oblateness, which dominate low-orbit error.

**How accurate are the pass times?**
Bounded by the TLE, not the code: error grows with time from epoch, especially for the ISS
(drag, reboosts). The tool reports TLE age, warns past 7 days and caps windows at 10. The
code itself is checked against a brute-force scan to 2 s.

**How do you know the pass finder is right?**
An independent brute-force 1 s scan agrees within tolerance (eval reference case). It shares
SGP4 with the code, so the next step would be a cross-check against an external predictor
(e.g. Heavens-Above) for the same TLE.

**Why didn't you apply the -0.1 offset the metadata declares?**
Because paired 03.00/05.00 scenes of the same day showed the pixels were already corrected
(DN medians within 9). Applying it would have pushed 53% of the AOI's red pixels negative.
I verify metadata against data when the cost of being wrong is silent.

**What does "valid fraction" mean and why report it?**
Share of AOI pixels clear (per SCL) in both scenes. Statistics over 20% of an AOI aren't
comparable to statistics over 100%; the user must know.

**Why read windows from COGs instead of downloading scenes?**
Only the bytes under the AOI move over the network; a 3.7 x 3.3 km NDVI comparison reads six
small windows in about 6 s instead of gigabytes.

**How do you make sure before/after pixels line up?**
Same MGRS tile required; AOI snapped to the 20 m grid; 20 m bands nearest-neighbour
upsampled to exact 2 x 2 blocks; a guard raises if a window is misaligned.

**How do you stop the model from making up numbers?**
Three layers: the system prompt forbids it; tools return pre-rounded values to quote; the
grounding eval flags any number, date or time not traceable to a tool result. It is a
heuristic (a correct paraphrase like "about half" isn't checked), so it pairs with
human review of live eval transcripts.

**What happens if Claude refuses?**
`stop_reason == "refusal"` is checked first and no tools run. The request opts into the
server-side fallback (`fallbacks="default"`), so a classifier false positive is retried on
Anthropic's recommended model in the same call; blocks before a `fallback` boundary are not
echoed into history.

**Why not force the tool call with tool_choice?**
On claude-opus-5-5 forced `tool_choice` returns a 400. Auto plus descriptions that say when
to call, plus strict schemas, plus an eval that checks tool selection.

**Why expose MCP as well as an agent?**
MCP makes the tools usable from Claude Code or Claude Desktop with no extra code; the
agent shows the same tools behind a custom loop with my own guardrails and evals.

**How did you test without network or an API key?**
Dependency-injected `DataContext` with recorded fixtures, synthetic labelled pixels, a fixed
clock and a JPL ephemeris excerpt; a socket guard in conftest; a fake Anthropic client that
scripts responses (refusal, max_tokens, parallel tool calls, fallback, typed errors).

**What would you change for production?**
OMM instead of TLE; async I/O and request coalescing; a shared cache (Redis/S3) respecting
CelesTrak policy; run `--live` evals on a schedule and track scores; structured logging and
tracing of tool calls; cross-validate pass predictions against an external source; larger
AOIs via mosaicking (odc-stac/stackstac) and cloud-free composites.

**Is an NDVI increase "more vegetation"?**
Not necessarily. NDVI responds to canopy cover, crop stage, irrigation, soil moisture,
sun angle and residual haze. orbitwatch reports what changed, not why.
