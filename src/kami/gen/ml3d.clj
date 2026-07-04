(ns kami.gen.ml3d
  "kami-gen-ml3d (ADR-2607051120): submits a `:model3d` generation job to the
  REAL `gftdcojp/cloud-murakumo` distributed GPU cloud (fn/engine :trellis,
  models trellis/hunyuan3d-2.1, :gpu/class :a100-80, fleet-auction placement)
  via `cloud-murakumo.gen/job` + `cloud-murakumo.worker/run-job` (not
  re-implemented -- see README 'cloud-murakumo dependency'), downloads the
  resulting glTF/GLB, auto-rigs it to a VRM humanoid via `kami.gen.ml3d.rig`
  (`kotoba-lang/skeleton` + `kotoba-lang/vrm`), and shapes a network-isekai
  Asset Hub `:asset/*` record for the result.

  IMPORTANT, read before calling `generate-from-image` with `real-execute`:
  no live GPU backend exists in this dev environment. `mock-execute` (a
  fixture glTF, `kami.gen.ml3d.fixture`) is what this repo's test suite and
  demonstration runs actually exercise end-to-end. `real-execute` is real,
  correctly wired code that WILL fail loudly if you call it without a
  configured backend (`MURAKUMO_BACKEND_URL`/`COMFY_URL`/`KAMI_RENDER_URL`) --
  it has never been run against a live TRELLIS/Hunyuan3D-2 endpoint. See
  README and ADR-2607051120."
  (:require [cloud-murakumo.spec :as spec]
            [cloud-murakumo.gen :as gen]
            [cloud-murakumo.worker :as worker]
            [cloud-murakumo.executor :as executor]
            [kami.gen.ml3d.rig :as rig]
            [kami.gen.ml3d.fixture :as fixture]
            [vrm :as vrm]
            [clojure.java.io :as io]))

(defn model3d-fn
  "The real `:model3d` `:gen` function declared in cloud-murakumo's
  `resources/murakumo.edn` (`:fn/engine :trellis`, models
  `trellis`/`hunyuan3d-2.1`, `:gpu/class :a100-80`, `:out [:gltf :glb]`).
  Loaded via `cloud-murakumo.spec/load-spec`, which reads
  `resources/murakumo.edn` off the classpath -- resolved here through the
  `:local/root` dependency on cloud-murakumo (its `resources` path lands on
  our classpath transitively; `deps.edn` comment + README explain why this
  live dependency was chosen over copying the function map as EDN)."
  []
  (or (first (filter #(= :model3d (:fn/id %)) (spec/functions (spec/load-spec))))
      (throw (ex-info "cloud-murakumo :model3d gen function not found in murakumo.edn (upstream schema changed?)" {}))))

(defn build-job
  "`{:model :prompt :refs :params}` + actor -> normalized `:gen.job` map, via
  `cloud-murakumo.gen/job` against the real `:model3d` function (not a
  reimplementation of job normalization -- `gen/job` validates modality/model
  and throws on an unsupported model, same as every other cloud-murakumo gen
  function)."
  [{:keys [model prompt refs params]} actor]
  (gen/job (model3d-fn) {:model model :prompt prompt :refs refs :params params} actor))

;; ── execute: mock (tested, used by this repo) vs real (wired, unexercised) ─

(defn mock-execute
  "Mock `execute` for this repo's test suite and dry-run demonstration.
  Ignores the invocation entirely and returns a freshly-generated fixture
  glTF/GLB (`kami.gen.ml3d.fixture`) plus a made-up `:gpu-seconds`. **This is
  not a TRELLIS/Hunyuan3D-2 result** -- see README."
  [_inv]
  {:outputs [(fixture/write-glb-file!)] :gpu-seconds 7})

(defn real-execute
  "Real `execute`: delegates to `cloud-murakumo.worker/default-execute`
  (which on JVM calls `cloud-murakumo.executor/execute` against the
  invocation's `:via`/`:backend` -- `:proc` shells out to the `trellis-cuda`
  image's CLI, `:http` calls a reachable service). This is genuinely the same
  code path cloud-murakumo's own worker uses; kami-gen-ml3d does not
  reimplement GPU execution.

  Adds one extra fail-closed guard on top of `default-execute` itself:
  cloud-murakumo's `:trellis` engine invocation carries no backend URL (it's
  `:via :proc`, not `:via :http`), so `default-execute` alone would happily
  attempt to spawn a `trellis-run` subprocess that doesn't exist on this
  machine and fail with a raw `IOException` -- correct, but not obviously
  \"no backend configured\" to a caller. This wrapper checks for ANY of the
  known backend env vars first and throws a clear, specific `ex-info` if none
  are set, matching cloud-murakumo's own `default-execute` fail-closed
  contract (`worker.cljc`: \"未配線時は明示エラー\") through this wrapper too,
  not just in cloud-murakumo itself. **Never actually invoked against a live
  backend in this repo** -- there isn't one in this dev environment
  (ADR-2607051120)."
  [inv]
  (when-not (or (worker/getenv "MURAKUMO_BACKEND_URL")
                (worker/getenv "COMFY_URL")
                (worker/getenv "KAMI_RENDER_URL"))
    (throw (ex-info (str "kami.gen.ml3d/real-execute: no live cloud-murakumo backend configured "
                         "(MURAKUMO_BACKEND_URL / COMFY_URL / KAMI_RENDER_URL all unset) -- "
                         "refusing to fabricate a result. Configure a real trellis-cuda backend "
                         "(fleet operator action, incurs real :gpu/class :a100-80 billing) or use "
                         "mock-execute for tests/dry-runs. See ADR-2607051120.")
                    {:via (:via inv) :runtime (get-in inv [:backend :runtime])})))
  (worker/default-execute inv))

(defn- capturing-execute
  "Wrap `execute` so we can retrieve its raw `{:outputs ...}` after
  `cloud-murakumo.worker/run-job` runs (that fn only returns CIDs, not the
  paths `cid-of` was computed from -- we need the actual mesh path for the
  auto-rig step)."
  [execute]
  (let [captured (atom nil)]
    [(fn [inv] (let [result (execute inv)] (reset! captured result) result))
     captured]))

;; ── Asset Hub publish shape (network-isekai public/assets/index.edn) ──────

(defn ->asset
  "Shape a completed run + exported VRM file path into a network-isekai Asset
  Hub `:asset/*` record (`:asset/kind :model3d`, CID-addressed payload --
  same `content-cid` sha256 convention as `cloud-murakumo.executor`, so this
  repo's CIDs are comparable to its 3 sibling `kami-gen-*` repos')."
  [done vrm-path]
  (let [cid (executor/content-cid vrm-path)
        model (:gen.job/model done)
        bytes-len (.length (io/file vrm-path))]
    {:asset/id (str "kami-gen-ml3d-" (get-in done [:gen.job/run :murakumo.run/id]))
     :asset/kind :model3d
     :asset/format :vrm
     :asset/title (str "kami-gen-ml3d auto-rig — " model)
     :asset/author "kotoba-lang/kami-gen-ml3d"
     :asset/license :cc0
     :asset/tags ["3d" "model3d" "vrm" "auto-rig" "trellis" model]
     :asset/payload {:cid {:hash cid} :bytes bytes-len :uri vrm-path}
     :asset/preview {:webgpu true}
     :asset/deps ["gftdcojp/cloud-murakumo" "kotoba-lang/vrm" "kotoba-lang/skeleton"]
     :asset/source :gen
     :asset/created (java.util.Date.)}))

;; ── the pipeline ──────────────────────────────────────────────────────────

(defn generate-from-image
  "`{:image-path :model :prompt :params :actor}` + `{:execute :cid-of
  :run-id}` -> `{:gen.job :run :mesh-path :vrm :asset}`.

  `:execute` is REQUIRED and must be injected explicitly (`mock-execute` or
  `real-execute`) -- there is no silent default, matching this org's
  fail-closed / no-silent-fallback convention (`cloud-murakumo.gen/resolve-
  model`'s own doc comment: 'silent fallback しない')."
  [{:keys [image-path model prompt params actor]
    :or {actor {:account "kami-gen-ml3d/dev"} params {}}}
   {:keys [execute cid-of run-id]
    :or {cid-of executor/content-cid
         run-id (fn [] (str "run-" (random-uuid)))}}]
  (when-not execute
    (throw (ex-info "generate-from-image requires :execute (kami.gen.ml3d/mock-execute or /real-execute)" {})))
  (let [refs (vec (remove nil? [image-path]))
        job (build-job {:model model :prompt prompt :refs refs :params params} actor)
        [wrapped-execute captured] (capturing-execute execute)
        done (worker/run-job job {:execute wrapped-execute :cid-of cid-of :run-id run-id})]
    (when (= :failed (:gen.job/status done))
      (throw (ex-info "kami-gen-ml3d: generation job failed" {:job done})))
    (let [mesh-path (first (:outputs @captured))
          vrm-bytes (rig/auto-rig-glb mesh-path)
          vrm-path (rig/write-vrm-file! vrm-bytes)
          vrm-document (vrm/parse-vrm vrm-bytes)
          asset (->asset done vrm-path)]
      {:gen.job job
       :run (:gen.job/run done)
       :mesh-path mesh-path
       :vrm {:path vrm-path :document vrm-document}
       :asset asset})))
