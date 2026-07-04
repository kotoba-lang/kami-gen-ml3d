(ns kami.gen.ml3d-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.java.io :as io]
            [cloud-murakumo.worker :as worker]
            [kami.gen.ml3d :as ml3d]
            [kami.gen.ml3d.rig :as rig]
            [kami.gen.ml3d.fixture :as fixture]
            [vrm :as vrm]
            [vrm.glb :as glb]))

(deftest model3d-fn-matches-cloud-murakumo-spec
  (testing "the real :model3d function from cloud-murakumo's murakumo.edn, not a copy"
    (let [f (ml3d/model3d-fn)]
      (is (= :model3d (:fn/id f)))
      (is (= :gen (:fn/kind f)))
      (is (= :trellis (:fn/engine f)))
      (is (= :3d (:fn/modality f)))
      (is (= :a100-80 (:gpu/class f)))
      (is (= #{"trellis" "hunyuan3d-2.1"} (set (keys (get-in f [:gen :models])))))
      (is (= "trellis" (get-in f [:gen :default])))
      (is (= [:gltf :glb] (get-in f [:gen :out]))))))

(deftest build-job-normalizes-per-cloud-murakumo-gen
  (let [job (ml3d/build-job {:model "trellis" :prompt nil :refs ["/tmp/ref.png"] :params {}}
                            {:account "test/demo"})]
    (is (= :3d (:gen.job/modality job)))
    (is (= :trellis (:gen.job/engine job)))
    (is (= :model3d (:gen.job/fn job)))
    (is (= "trellis" (:gen.job/model job)))
    (is (= :queued (:gen.job/status job)))
    (is (= ["/tmp/ref.png"] (get-in job [:gen.job/input :refs])))))

(deftest build-job-rejects-unknown-model
  (testing "unsupported model throws -- no silent fallback (cloud-murakumo.gen/resolve-model contract)"
    (is (thrown? clojure.lang.ExceptionInfo
                 (ml3d/build-job {:model "dall-e" :refs ["/tmp/ref.png"]} {:account "test"})))))

(deftest mock-execute-returns-fixture-glb
  (let [{:keys [outputs gpu-seconds]} (ml3d/mock-execute {:via :proc :backend {:runtime "TRELLIS"}})]
    (is (= 1 (count outputs)))
    (is (.exists (io/file (first outputs))))
    (is (pos? gpu-seconds))
    ;; structurally valid GLB (parses without throwing)
    (let [raw (vec (map #(bit-and (int %) 0xFF) (.readAllBytes (io/input-stream (first outputs)))))]
      (is (map? (glb/parse-glb raw))))))

(deftest real-execute-fails-closed-without-backend
  (testing "matches cloud-murakumo's own default-execute fail-closed contract, through this wrapper"
    (with-redefs [worker/getenv (fn [_] nil)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"no live cloud-murakumo backend configured"
                            (ml3d/real-execute {:via :proc :backend {:runtime "TRELLIS"}}))))))

(defn- approx= [a b] (every? true? (map (fn [x y] (< (Math/abs (- (double x) (double y))) 1e-4)) a b)))

(deftest rig-bounding-box-matches-fixture
  (testing "float32 round-trip through the GLB binary chunk -- approx-equal, not exact"
    (let [doc (rig/read-glb-doc (fixture/write-glb-file!))
          bbox (rig/bounding-box doc)]
      (is (approx= fixture/box-min (:min bbox)))
      (is (approx= fixture/box-max (:max bbox))))))

(deftest rig-build-skeleton-shape
  (let [bbox {:min [-0.3 0.0 -0.2] :max [0.3 1.8 0.2]}
        {:keys [sk names name->index]} (rig/build-skeleton bbox)]
    (is (= 19 (count names)))
    (is (= 0 (name->index "hips")))
    (is (nil? (:parent (first (:bones sk)))))
    (is (every? some? (map :parent (rest (:bones sk)))))))

(deftest auto-rig-glb-round-trips-through-vrm-parse
  (testing "the exported VRM GLB is a real, valid VRM 1.0 file per kotoba-lang/vrm's own parser"
    (let [mesh-path (fixture/write-glb-file!)
          vrm-bytes (rig/auto-rig-glb mesh-path)
          doc (vrm/parse-vrm vrm-bytes)]
      (is (= :v1-0 (:version doc)))
      (is (= 19 (count (:human-bones (:humanoid doc)))))
      (testing "vrm.humanoid/to-kami-skeleton re-reads the same skin/skeleton we wrote"
        (let [sk (vrm/to-kami-skeleton doc)]
          (is (= 19 (count (:bones sk))))
          (is (= "hips" (:name (first (:bones sk)))))))
      (testing "mesh primitive carries real skin weight attributes"
        (let [prim (get-in doc [:gltf :meshes 0 :primitives 0])]
          (is (contains? (:attributes prim) :JOINTS_0))
          (is (contains? (:attributes prim) :WEIGHTS_0)))))))

(deftest generate-from-image-end-to-end-against-mock-executor
  (testing "full pipeline: job -> run -> mesh -> auto-rig -> asset, all against the mock executor"
    (let [result (ml3d/generate-from-image
                  {:image-path "/tmp/reference-photo.png" :model "trellis"}
                  {:execute ml3d/mock-execute})]
      (testing ":gen.job shape (cloud-murakumo.gen/job normalization)"
        (is (= :trellis (get-in result [:gen.job :gen.job/engine])))
        (is (= "trellis" (get-in result [:gen.job :gen.job/model]))))
      (testing ":run shape (:murakumo.run ledger entry, cloud-murakumo.worker/run-job)"
        (is (= :done (:murakumo.run/state (:run result))))
        (is (= 7 (:murakumo.run/gpu-seconds (:run result))))
        (is (= :3d (:murakumo.run/modality (:run result)))))
      (testing ":mesh-path is a real downloaded (fixture) file"
        (is (.exists (io/file (:mesh-path result)))))
      (testing ":vrm is a real, parseable VRM document"
        (is (.exists (io/file (get-in result [:vrm :path]))))
        (is (= 19 (count (:human-bones (:humanoid (get-in result [:vrm :document])))))))
      (testing ":asset matches the network-isekai Asset Hub :asset/* schema"
        (let [asset (:asset result)]
          (is (= :model3d (:asset/kind asset)))
          (is (= :vrm (:asset/format asset)))
          (is (re-find #"^bafy-sha256-" (get-in asset [:asset/payload :cid :hash])))
          (is (pos? (get-in asset [:asset/payload :bytes])))
          (is (= (get-in result [:vrm :path]) (get-in asset [:asset/payload :uri])))
          (is (contains? (set (:asset/deps asset)) "gftdcojp/cloud-murakumo")))))))

(deftest generate-from-image-requires-execute
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires :execute"
                        (ml3d/generate-from-image {:image-path "/tmp/x.png"} {}))))
